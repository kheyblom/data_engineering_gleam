"""Verify a finished GLEAM icechunk zarr store against the raw netCDF files.

Reads the same config the build did, so it checks the store the config asks
for and needs no paths of its own. Nothing here writes to the store: every
session is a read-only session, and the one destructive operation it touches
(``garbage_collect``) is only ever called as a dry run.

The raw side is read with ``netCDF4`` and masking switched **off**, so the
fill sentinel arrives as the literal -999 the files hold. That is deliberate:
comparing against ``xr.open_dataset`` would send both sides through the same
CF decoding path and could only prove the pipeline agrees with itself, not
that -999 became NaN. Every value comparison is exact rather than
``allclose`` -- the pipeline does no arithmetic, so anything short of
bit-identical is a defect.

Six phases, selectable with ``--phases``:

``structure``
    Dimensions, chunk shapes, dtypes, fill values, codecs and attributes,
    read from the zarr metadata alone.
``index``
    That all variables share one time axis, and that the store's time, lat
    and lon are bit-identical to the raw files'.
``samples``
    Full lat/lon planes at random and at chosen timesteps, compared cell by
    cell against raw. Stratified so every variable is covered.
``sweep``
    Every chunk in the store, via the manifest rather than by reading data:
    presence, placement, and compressed size. An absent chunk is checked
    against raw before it counts as a finding, since zarr does not write a
    chunk whose cells are all fill (see ``check_sweep``).
``identities``
    GLEAM's own component sums. These can only close if the variables were
    merged onto the right grid cell at the right timestep, which makes them
    the cheapest available check on the merge itself.
``ranges``
    Physical bounds implied by each variable's ``units`` attribute.

A phase reports FAIL only for something the *store* got wrong. Properties of
the upstream data that look like defects -- E's all-fill days, a valid-cell
footprint that moves day to day, Ep's capped cells -- are confirmed against
raw and reported as notes, because asserting them as invariants produces
false alarms rather than findings. The history of each is in TESTING.md.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import logging
import os
import sys

import icechunk
import netCDF4
import numpy as np
import xarray as xr
import zarr

from utils.log_utils import setup_logging
from utils.path_utils import (
    load_config,
    resolve_variables,
    store_path,
    variable_files,
)
from utils.zarr_utils import BRANCH, DEFAULT_TIMESTEPS_PER_COMMIT

LOG = logging.getLogger(__name__)

# the sentinel the raw files carry; build_encoding replaces it with NaN
RAW_FILL = np.float32(-999.0)

# plausible bounds keyed on the units attribute, so a new variable inherits
# them rather than needing a table entry of its own
BOUNDS_BY_UNITS = {
    'mm.day-1': (-50.0, 200.0),     # evaporation fluxes, either sign
    'W.m-2': (-1000.0, 1000.0),     # sensible heat flux
    'm3.m-3': (0.0, 1.0),           # volumetric soil moisture
    '-': (0.0, 1.0),                # evaporative stress, dimensionless
}

# GLEAM's actual evaporation is the sum of its components, with condensation
# entering negative; potential evaporation splits into two components. Both
# hold to float32 rounding, so they pin the merge rather than merely sanity
# check it.
IDENTITIES = (
    ('E', ('Eb', 'Ec', 'Ei', 'Es', 'Et', 'Ew')),
    ('Ep', ('Ep_aero', 'Ep_rad')),
)

# a float32 plane of real geophysical data never zstds this small; a constant
# or truncated plane lands orders of magnitude under it
DEGENERATE_BYTES = 50 * 1024

PHASES = ('structure', 'index', 'samples', 'sweep', 'identities', 'ranges')


class Report:
    """Collects check outcomes so the exit status can reflect all of them.

    Attributes:
        failures (list): Labels of the checks that failed.
        n_checks (int): How many checks were recorded.
    """

    def __init__(self):
        self.failures = []
        self.n_checks = 0

    def check(self, label, ok, detail=''):
        """Record a pass or fail.

        Args:
            label (str): What was checked.
            ok (bool): Whether it held.
            detail (str): Measured values to log alongside the verdict.

        Returns:
            bool: The ``ok`` that was passed in, so callers can branch on it.
        """
        self.n_checks += 1
        LOG.info(f'{"PASS" if ok else "FAIL"}  {label}{"  " + detail if detail else ""}')
        if not ok:
            self.failures.append(label)
        return ok

    def note(self, message):
        """Log an observation that is not a pass/fail judgement.

        Args:
            message (str): The observation.
        """
        LOG.info(f'note  {message}')


def parse_args():
    """Parse the command line.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='verify a built gleam zarr store against the raw netcdf files.'
    )
    parser.add_argument(
        '--config', type=str, required=True, help='Path to YAML configuration file.'
    )
    parser.add_argument(
        '--phases',
        type=str,
        default=','.join(PHASES),
        help=f'Comma separated subset of {",".join(PHASES)}.',
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=42,
        help='Random planes to compare, spread evenly over the variables.',
    )
    parser.add_argument(
        '--seed', type=int, default=20260910, help='Seed for the sample draw.'
    )
    parser.add_argument(
        '--identity-days',
        type=int,
        default=4,
        help='Random days on which to test the component identities.',
    )
    parser.add_argument(
        '--range-days',
        type=int,
        default=6,
        help='Random days per variable for the bounds check.',
    )
    return parser.parse_args()


def open_store(settings):
    """Open the configured store read only.

    Args:
        settings (dict): The loaded configuration.

    Returns:
        tuple: The icechunk.Repository, its read-only Session, and the
            decoded xarray.Dataset.
    """
    path = store_path(settings)
    LOG.info(f'verifying {path}')
    repository = icechunk.Repository.open(icechunk.local_filesystem_storage(path))
    session = repository.readonly_session(branch=BRANCH)
    return repository, session, xr.open_zarr(session.store, consolidated=False)


def build_raw_index(settings, variables):
    """Index the raw files so any store timestep can be traced back to one.

    Reads only each file's time dimension and time values, which is a metadata
    read; the planes themselves are read lazily later.

    Args:
        settings (dict): The loaded configuration.
        variables (list): Variable names to index.

    Returns:
        dict: Variable name -> dict with 'files', 'lengths', 'offsets' (the
            cumulative first global timestep of each file) and 'time'.
    """
    index = {}
    for name in variables:
        files = variable_files(settings, name)
        lengths, times = [], []
        for path in files:
            dataset = netCDF4.Dataset(path)
            lengths.append(dataset.dimensions['time'].size)
            times.append(np.asarray(dataset.variables['time'][:]))
            dataset.close()
        index[name] = {
            'files': files,
            'lengths': np.asarray(lengths),
            'offsets': np.concatenate([[0], np.cumsum(lengths)]),
            'time': np.concatenate(times),
        }
        LOG.info(
            f'indexed {name:9s} {len(files)} files, '
            f'{index[name]["offsets"][-1]} timesteps'
        )
    return index


def locate(index, name, timestep):
    """Map a global store timestep onto a raw file and an index within it.

    Args:
        index (dict): The raw index from ``build_raw_index``.
        name (str): Variable name.
        timestep (int): Global timestep, as indexed in the store.

    Returns:
        tuple: The file path and the timestep's index inside that file.
    """
    meta = index[name]
    which = int(np.searchsorted(meta['offsets'], timestep, side='right') - 1)
    return meta['files'][which], int(timestep - meta['offsets'][which])


def read_raw_plane(path, name, local):
    """Read one lat/lon plane from a raw file with the fill sentinel intact.

    Args:
        path (str): The netCDF file to read.
        name (str): Variable to read.
        local (int): Timestep index within the file.

    Returns:
        numpy.ndarray: The float32 plane, still holding -999 where masked.
    """
    dataset = netCDF4.Dataset(path)
    # masking and scaling off, so the comparison does not run through the same
    # CF decoding the pipeline used and cannot agree with it by construction
    dataset.set_auto_mask(False)
    dataset.set_auto_scale(False)
    plane = np.asarray(dataset.variables[name][local]).astype(np.float32)
    dataset.close()
    return plane


def compare_plane(stored, plane_raw):
    """Compare a stored plane against its raw counterpart.

    Args:
        stored (numpy.ndarray): The decoded plane from the store.
        plane_raw (numpy.ndarray): The raw plane, fill sentinel intact.

    Returns:
        tuple: Whether the NaN pattern matches the fill pattern, whether every
            non-fill cell is bit-identical, and the count of valid cells.
    """
    is_fill = plane_raw == RAW_FILL
    mask_ok = np.array_equal(is_fill, np.isnan(stored))
    values_ok = bool(np.all(stored[~is_fill] == plane_raw[~is_fill]))
    return mask_ok, values_ok, int((~is_fill).sum())


def check_structure(report, session, dataset, variables, settings):
    """Check dimensions, chunking, dtypes, fill values and attributes.

    Reads zarr metadata only, so this is cheap enough to run every time.

    Args:
        report (Report): Where to record outcomes.
        session (icechunk.Session): Read-only session on the store.
        dataset (xarray.Dataset): The decoded store.
        variables (list): Variable names the config asks for.
        settings (dict): The loaded configuration.
    """
    root = zarr.open_group(session.store, mode='r')
    n_time = dataset.sizes['time']

    report.check(
        'store holds exactly the configured variables',
        set(dataset.data_vars) == set(variables),
        f'{len(dataset.data_vars)} variables',
    )
    report.note(f'dimensions {dict(dataset.sizes)}, {dataset.nbytes / 1024**4:.2f} TiB logical')

    # the chunk shape the config asks for, with -1 meaning the whole dimension
    expected = tuple(
        dataset.sizes[dimension] if size == -1 else size
        for dimension, size in settings['chunks'].items()
    )
    for name in sorted(dataset.data_vars):
        array = root[name]
        report.check(
            f'{name:9s} chunks {expected}, float32, NaN fill, dims (time, lat, lon)',
            array.chunks == expected
            and array.dtype == np.float32
            and np.isnan(array.fill_value)
            and dataset[name].dims == ('time', 'lat', 'lon'),
            f'chunks={array.chunks} dtype={array.dtype} fill={array.fill_value!r}',
        )
        report.check(
            f'{name:9s} carries its CF attributes',
            {'standard_name', 'long_name', 'units'} <= set(dataset[name].attrs),
            str(sorted(dataset[name].attrs)),
        )

    # the time coordinate is chunked one commit batch per chunk, which is what
    # kept every append landing on a chunk boundary
    batch = settings.get('timesteps_per_commit', DEFAULT_TIMESTEPS_PER_COMMIT)
    report.check(
        f'time coordinate chunked one commit batch ({batch}) per chunk',
        root['time'].chunks == (batch,),
        str(root['time'].chunks),
    )
    encoding = dataset['time'].encoding
    report.check(
        'time encoding pinned to int64 days since 1900-01-01, proleptic_gregorian',
        encoding.get('units') == 'days since 1900-01-01'
        and encoding.get('calendar') == 'proleptic_gregorian'
        and np.dtype(encoding.get('dtype')) == np.int64,
        f'{encoding.get("units")!r} {encoding.get("calendar")!r} {encoding.get("dtype")}',
    )
    for coordinate in ('lat', 'lon'):
        report.check(
            f'{coordinate} coordinate is a single chunk',
            root[coordinate].chunks == (dataset.sizes[coordinate],),
            str(root[coordinate].chunks),
        )
    report.note(f'codecs {root[variables[0]].metadata.codecs}')
    report.note(f'global attributes {sorted(dataset.attrs)}')

    # a daily store must have one timestep per day of its own span, or a file
    # was missed somewhere between the download and the merge
    days = dataset['time'].values
    span = int((days[-1] - days[0]) / np.timedelta64(1, 'D')) + 1
    report.check(
        'one timestep per day across the whole span, monotonic and unique',
        span == n_time
        and bool(np.all(np.diff(days) == np.timedelta64(1, 'D'))),
        f'{n_time} timesteps, {str(days[0])[:10]} .. {str(days[-1])[:10]} ({span} days)',
    )


def check_index(report, dataset, index, variables):
    """Check the time axis agreement and the coordinates, against raw.

    Args:
        report (Report): Where to record outcomes.
        dataset (xarray.Dataset): The decoded store.
        index (dict): The raw index.
        variables (list): Variable names.
    """
    reference = index[variables[0]]['time']
    # merge_variables already refuses a mismatch here, but it refuses at build
    # time; this confirms the store on disk was built from an agreeing set
    mismatched = [
        name
        for name in variables[1:]
        if not np.array_equal(index[name]['time'], reference)
    ]
    report.check(
        f'all {len(variables)} raw variables share one time axis',
        not mismatched,
        f'mismatched: {mismatched}' if mismatched else f'{len(reference)} timesteps',
    )

    # compare the integers actually on disk, not the decoded datetimes, so the
    # pinned encoding is part of what is being checked
    stored_days = (
        (dataset['time'].values - np.datetime64('1900-01-01'))
        / np.timedelta64(1, 'D')
    ).astype(np.int64)
    report.check(
        'store time bit-identical to raw days since 1900',
        len(stored_days) == len(reference) and np.array_equal(stored_days, reference),
        f'{len(stored_days)} vs {len(reference)} timesteps',
    )

    path, _ = locate(index, variables[0], 0)
    raw = netCDF4.Dataset(path)
    raw.set_auto_mask(False)
    for coordinate in ('lat', 'lon'):
        report.check(
            f'{coordinate} bit-identical to raw',
            np.array_equal(
                dataset[coordinate].values, np.asarray(raw.variables[coordinate][:])
            ),
            f'{dataset[coordinate].values[0]:.4f} .. {dataset[coordinate].values[-1]:.4f}',
        )
    raw.close()


def sample_timesteps(dataset, index, variables, args):
    """Choose which (variable, timestep) planes to compare.

    Draws are stratified over the variables so a single mis-merged variable
    cannot hide behind a lucky draw, then the batch boundaries and the file
    seams are added deliberately, since those are where an append or a resume
    would go wrong and random draws rarely land on them.

    Args:
        dataset (xarray.Dataset): The decoded store.
        index (dict): The raw index.
        variables (list): Variable names.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        list: (variable, timestep) pairs to compare.
    """
    rng = np.random.default_rng(args.seed)
    n_time = dataset.sizes['time']

    per_variable = max(1, args.samples // len(variables))
    picks = [
        (name, int(rng.integers(n_time)))
        for name in variables
        for _ in range(per_variable)
    ]

    # both sides of every yearly file seam, rotating the variable so the seams
    # spread over the variables too
    offsets = index[variables[0]]['offsets']
    for i, seam in enumerate(offsets[1:-1]):
        name = variables[i % len(variables)]
        picks.append((name, int(seam) - 1))
        picks.append((name, int(seam)))

    # the first and last timestep, and either side of the batch boundaries
    boundaries = {0, n_time - 1, n_time - 2}
    for edge in (100, 8200, 15300):
        boundaries.update({edge - 1, edge, edge + 1})
    for timestep in sorted(boundaries):
        if 0 <= timestep < n_time:
            picks.append((variables[int(rng.integers(len(variables)))], timestep))
    return picks


def check_samples(report, dataset, index, variables, args):
    """Compare full lat/lon planes against raw, cell by cell.

    Args:
        report (Report): Where to record outcomes.
        dataset (xarray.Dataset): The decoded store.
        index (dict): The raw index.
        variables (list): Variable names.
        args (argparse.Namespace): Parsed arguments.
    """
    picks = sample_timesteps(dataset, index, variables, args)
    dates = dataset['time'].values
    LOG.info(f'comparing {len(picks)} full planes against raw')

    n_failed = 0
    for name, timestep in picks:
        path, local = locate(index, name, timestep)
        plane_raw = read_raw_plane(path, name, local)
        stored = dataset[name].isel(time=timestep).values
        mask_ok, values_ok, n_valid = compare_plane(stored, plane_raw)

        # the store's own time value must be the one the raw file carries at
        # that index, or the plane is right but landed on the wrong day
        raw = netCDF4.Dataset(path)
        raw_day = int(np.asarray(raw.variables['time'][local]))
        raw.close()
        stored_day = int((dates[timestep] - np.datetime64('1900-01-01')) / np.timedelta64(1, 'D'))

        if not (mask_ok and values_ok and stored_day == raw_day):
            n_failed += 1
            report.check(
                f'{name:9s} t={timestep} ({str(dates[timestep])[:10]}) matches raw',
                False,
                f'mask={mask_ok} values={values_ok} day={stored_day == raw_day}',
            )

    report.check(
        f'all {len(picks)} sampled planes bit-identical to raw, on the right day',
        n_failed == 0,
        f'{len(picks) - n_failed} clean, covering all {len(variables)} variables '
        f'and all {len(index[variables[0]]["files"]) - 1} file seams',
    )


async def check_sweep(report, session, dataset, index, variables):
    """Check every chunk in the store for presence, placement and size.

    Runs off the manifest rather than by reading data, so it covers the whole
    store in seconds. An absent chunk is not assumed to be a defect: zarr does
    not write a chunk whose cells are all equal to the fill value, so an
    all-fill raw plane correctly leaves a hole that reads back as NaN. Each
    hole is therefore traced to raw and only counts as a finding if raw holds
    real data there.

    Args:
        report (Report): Where to record outcomes.
        session (icechunk.Session): Read-only session on the store.
        dataset (xarray.Dataset): The decoded store.
        index (dict): The raw index.
        variables (list): Variable names.
    """
    n_time = dataset.sizes['time']
    dates = dataset['time'].values
    total_bytes = 0

    for name in variables:
        coordinates = [c async for c in session.chunk_coordinates(f'/{name}')]
        present = {c[0] for c in coordinates}
        report.check(
            f'{name:9s} every chunk on the time axis, none misplaced',
            len(coordinates) == len(present)
            and all(c[1] == 0 and c[2] == 0 for c in coordinates),
            f'{len(coordinates)} chunks',
        )

        sizes = np.asarray(
            await asyncio.gather(
                *(session.store.getsize(f'{name}/c/{t}/0/0') for t in sorted(present))
            )
        )
        total_bytes += int(sizes.sum())
        degenerate = [
            sorted(present)[i] for i in np.flatnonzero(sizes < DEGENERATE_BYTES)
        ]
        report.check(
            f'{name:9s} no written chunk is degenerate or truncated',
            not degenerate,
            f'compressed min={sizes.min() / 1024**2:.2f} '
            f'median={np.median(sizes) / 1024**2:.2f} '
            f'max={sizes.max() / 1024**2:.2f} MiB'
            + (f', degenerate at {degenerate[:5]}' if degenerate else ''),
        )

        # a hole is legitimate only where the raw plane is entirely fill
        missing = sorted(set(range(n_time)) - present)
        if not missing:
            continue
        real_loss = []
        for timestep in missing:
            path, local = locate(index, name, timestep)
            plane_raw = read_raw_plane(path, name, local)
            if not bool(np.all(plane_raw == RAW_FILL)):
                real_loss.append(timestep)
        report.check(
            f'{name:9s} every absent chunk is an all-fill raw plane',
            not real_loss,
            f'{len(missing)} absent, raw all-fill at every one'
            if not real_loss
            else f'DATA LOSS at {real_loss[:5]}',
        )
        report.note(
            f'{name} has no chunk on {len(missing)} days because raw is entirely '
            f'fill there: {", ".join(str(dates[t])[:10] for t in missing)}'
        )

    report.note(
        f'{total_bytes / 1024**4:.3f} TiB compressed across the data chunks, '
        f'{total_bytes / dataset.nbytes:.3f} of logical'
    )

    time_chunks = [c async for c in session.chunk_coordinates('/time')]
    expected = -(-n_time // dataset['time'].encoding['chunks'][0])
    report.check(
        'time coordinate chunks contiguous and complete',
        sorted(c[0] for c in time_chunks) == list(range(expected)),
        f'{len(time_chunks)} of {expected}',
    )


def check_identities(report, dataset, index, args):
    """Check GLEAM's component sums, against raw where they do not close.

    A sum that closes to float32 rounding can only do so if every component
    was merged onto the same grid cell at the same timestep, which is why this
    is worth more than its cost. Where it does not close, the residual field
    is recomputed from raw: if the two residual fields are identical the store
    reproduced the upstream data faithfully and the gap is GLEAM's, not the
    pipeline's.

    Args:
        report (Report): Where to record outcomes.
        dataset (xarray.Dataset): The decoded store.
        index (dict): The raw index.
        args (argparse.Namespace): Parsed arguments.
    """
    rng = np.random.default_rng(args.seed)
    dates = dataset['time'].values
    days = [int(x) for x in rng.integers(dataset.sizes['time'], size=args.identity_days)]

    for total_name, components in IDENTITIES:
        if total_name not in dataset.data_vars or not all(
            c in dataset.data_vars for c in components
        ):
            continue
        for timestep in days:
            total = dataset[total_name].isel(time=timestep).values
            if not np.isfinite(total).any():
                # an all-fill day for the total, so there is nothing to close
                continue
            summed = sum(
                dataset[name].isel(time=timestep).values for name in components
            )
            residual = np.abs(total - summed)
            valid = np.isfinite(total) & np.isfinite(summed)
            worst = float(np.max(residual[valid])) if valid.any() else 0.0
            label = f'{total_name} == {" + ".join(components)} at {str(dates[timestep])[:10]}'

            if worst < 1e-4:
                report.check(f'{label} closes to float32 rounding', True,
                             f'max|residual|={worst:.3e}')
                continue

            # recompute the identity from raw, decoded the same way, and see
            # whether the store's residual field is the raw one exactly
            raw_planes = {}
            for name in (total_name, *components):
                path, local = locate(index, name, timestep)
                plane = read_raw_plane(path, name, local)
                plane[plane == RAW_FILL] = np.nan
                raw_planes[name] = plane
            raw_residual = np.abs(
                raw_planes[total_name] - sum(raw_planes[n] for n in components)
            )
            same = np.array_equal(
                np.nan_to_num(residual, nan=-1.0), np.nan_to_num(raw_residual, nan=-1.0)
            )
            n_over = int((np.nan_to_num(residual, nan=0.0) > 1e-4).sum())
            report.check(
                f'{label} does not close, and the store reproduces raw exactly there',
                same,
                f'{n_over} of {int(valid.sum())} cells over 1e-4, '
                f'max|residual|={worst:.3e}, identical to raw={same}',
            )
            report.note(
                f'{label}: upstream GLEAM does not close on {n_over} cells '
                f'({100 * n_over / max(1, int(valid.sum())):.4f}%); the store matches raw'
            )


def check_ranges(report, dataset, index, variables, args):
    """Check physical bounds, and report how the valid footprint moves.

    The bounds are asserted. The valid-cell footprint is only reported: in
    GLEAM v4.3a it genuinely moves from day to day for most variables, and the
    plane comparisons already prove the NaN pattern matches raw exactly, so
    asserting a static mask would report upstream behaviour as a defect.

    Args:
        report (Report): Where to record outcomes.
        dataset (xarray.Dataset): The decoded store.
        index (dict): The raw index.
        variables (list): Variable names.
        args (argparse.Namespace): Parsed arguments.
    """
    rng = np.random.default_rng(args.seed)
    days = sorted(
        set(int(x) for x in rng.integers(dataset.sizes['time'], size=args.range_days))
    )

    for name in variables:
        units = dataset[name].attrs.get('units')
        bounds = BOUNDS_BY_UNITS.get(units)
        counts, lows, highs = [], [], []
        for timestep in days:
            values = dataset[name].isel(time=timestep).values
            finite = np.isfinite(values)
            counts.append(int(finite.sum()))
            if finite.any():
                lows.append(float(np.min(values[finite])))
                highs.append(float(np.max(values[finite])))

        if bounds is None:
            report.note(f'{name} has units {units!r} with no bounds to check')
            continue
        low, high = bounds
        observed = (min(lows), max(highs)) if lows else (np.nan, np.nan)
        report.check(
            f'{name:9s} within [{low:g}, {high:g}] {units}',
            not lows or (observed[0] >= low and observed[1] <= high),
            f'observed [{observed[0]:.4g}, {observed[1]:.4g}]',
        )

        # a moving footprint is upstream; confirm the store follows raw on the
        # day it is widest apart rather than asserting it stays put
        if len(set(counts)) > 1:
            timestep = days[int(np.argmin(counts))]
            path, local = locate(index, name, timestep)
            plane_raw = read_raw_plane(path, name, local)
            stored = dataset[name].isel(time=timestep).values
            mask_ok, _, n_valid = compare_plane(stored, plane_raw)
            report.check(
                f'{name:9s} moving valid footprint follows raw exactly',
                mask_ok,
                f'valid cells {min(counts)}..{max(counts)} over {len(days)} days, '
                f'{n_valid} at {str(dataset["time"].values[timestep])[:10]}',
            )
        else:
            report.note(f'{name} valid footprint is static at {counts[0]} cells')


def check_repository(report, repository, dataset, settings):
    """Report the commit history and the unreachable objects.

    The garbage collection call is a dry run, which reads and reports without
    deleting anything.

    Args:
        report (Report): Where to record outcomes.
        repository (icechunk.Repository): The repository holding the store.
        dataset (xarray.Dataset): The decoded store.
        settings (dict): The loaded configuration.
    """
    snapshots = list(repository.ancestry(branch=BRANCH))
    batch = settings.get('timesteps_per_commit', DEFAULT_TIMESTEPS_PER_COMMIT)
    # one commit per batch, plus the repository's own initial snapshot
    expected = -(-dataset.sizes['time'] // batch) + 1
    # a finalization commit -- provenance attributes, for instance -- legitimately
    # adds to this, so the build's batch count is a floor rather than an equality
    report.check(
        'at least one reachable snapshot per commit batch, plus the initial one',
        len(snapshots) >= expected,
        f'{len(snapshots)} of {expected} expected from the build',
    )
    # ancestry yields newest first, so any surplus is the head of the list
    if len(snapshots) > expected:
        messages = [s.message for s in snapshots[: len(snapshots) - expected]]
        report.note(
            f'{len(snapshots) - expected} snapshot(s) beyond the build: {messages}'
        )

    summary = repository.garbage_collect(
        datetime.datetime.now(datetime.timezone.utc), dry_run=True
    )
    report.note(
        f'unreachable objects (dry run, nothing deleted): '
        f'{summary.bytes_deleted / 1024**3:.2f} GiB, '
        f'{summary.chunks_deleted} chunks, {summary.snapshots_deleted} snapshots'
    )


def main(settings, args):
    """Run the selected phases and return a process exit status.

    Args:
        settings (dict): The loaded configuration.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: 0 if every check passed, 1 otherwise.

    Raises:
        ValueError: If --phases names something that is not a phase.
    """
    phases = [p.strip() for p in args.phases.split(',') if p.strip()]
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        raise ValueError(f'unknown phases {unknown}; choose from {list(PHASES)}')

    report = Report()
    variables = resolve_variables(settings)
    repository, session, dataset = open_store(settings)
    LOG.info(f'phases: {phases}')

    # every phase but 'structure' traces store values back to a raw file
    index = (
        build_raw_index(settings, variables)
        if set(phases) - {'structure'}
        else {}
    )

    if 'structure' in phases:
        LOG.info('--- structure')
        check_structure(report, session, dataset, variables, settings)
        check_repository(report, repository, dataset, settings)
    if 'index' in phases:
        LOG.info('--- index')
        check_index(report, dataset, index, variables)
    if 'samples' in phases:
        LOG.info('--- samples')
        check_samples(report, dataset, index, variables, args)
    if 'sweep' in phases:
        LOG.info('--- sweep')
        asyncio.run(check_sweep(report, session, dataset, index, variables))
    if 'identities' in phases:
        LOG.info('--- identities')
        check_identities(report, dataset, index, args)
    if 'ranges' in phases:
        LOG.info('--- ranges')
        check_ranges(report, dataset, index, variables, args)

    LOG.info(f'{report.n_checks} checks, {len(report.failures)} failures')
    if report.failures:
        for label in report.failures:
            LOG.error(f'failed: {label}')
        return 1
    LOG.info('store verified :-)')
    return 0


if __name__ == '__main__':
    arguments = parse_args()
    configuration = load_config(arguments.config)
    setup_logging(
        os.path.join(configuration['directories']['logs'], 'verify_gleam_zarr.log')
    )
    sys.exit(main(configuration, arguments))
