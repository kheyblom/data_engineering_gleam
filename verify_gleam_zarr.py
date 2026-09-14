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

The store's chunking decides what a cheap read is, and the phases that read
data follow it rather than assuming one layout. The unit of comparison is a
*box* -- a (time, lat, lon) range that is one chunk of the store being checked:

``spatial``  (chunks ``1 x 1800 x 3600``) a box is one whole global plane on
    one day.
``temporal`` (chunks ``16802 x 20 x 20``) a box is one small lat/lon tile
    through the entire record.

Reading the wrong one is not a style question: a single global plane out of a
time-chunked store touches every chunk of the variable, some 400 GiB, to return
one map. The box shape is picked once, in ``box_shape``, and everything
downstream follows it. Either way a box is read from the store and from the raw
files independently and compared cell by cell.

Seven phases, selectable with ``--phases``. The first six run by default; the
seventh needs a second store to compare against:

``structure``
    Dimensions, chunk shapes, dtypes, fill values, codecs and attributes,
    read from the zarr metadata alone.
``index``
    That all variables share one time axis, and that the store's time, lat
    and lon are bit-identical to the raw files'.
``samples``
    Boxes drawn at random and at chosen positions, compared cell by cell
    against raw. Stratified so every variable is covered.
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
``metadata``
    The store's attributes and coordinates against a sibling store's, so two
    stores built from the same files describe the same data the same way.
    Needs ``--compare-with``, so it is not part of a default run.

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
from utils.zarr_utils import (
    BRANCH,
    commit_batch_size,
    iter_blocks,
    resolve_block_shape,
    resolve_chunks,
)

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

# the floor a written chunk's compressed size may not fall below. Zarr only
# writes a chunk holding something other than fill, so a small chunk is not by
# itself wrong -- what it cannot be is empty or truncated.
#
# How much a floor is worth depends on the layout, because the smallest
# legitimate chunk does. A global float32 plane of real geophysical data never
# zstds under 50 KiB, so that floor catches a constant or truncated plane with
# no false alarms. A 20x20 tile is a different matter: the smallest legitimate
# one holds a single valid cell on a single day of 46 years, a few hundred
# bytes of signal in an ocean of NaN, so any floor above zero would report
# sparse coastal tiles as damage. There the floor is only 'not empty', and the
# real anti-truncation check is to read the smallest chunks back and compare
# them against raw -- measured rather than assumed.
DEGENERATE_BYTES = {'spatial': 50 * 1024, 'temporal': 0}

PHASES = (
    'structure', 'index', 'samples', 'sweep', 'identities', 'ranges', 'metadata'
)

# 'metadata' compares two stores, so it needs a second config and cannot run in
# a default sweep of a single one
DEFAULT_PHASES = tuple(p for p in PHASES if p != 'metadata')

# global attributes that describe how a store is laid out, when it was built or
# what has been proven about it. Two stores built from the same files
# legitimately differ on these and must agree on everything else: the rest is
# either carried over from the source files or shared prose, so a difference
# there means the two stores describe the same data differently.
LAYOUT_ATTRS = frozenset(
    {
        'title',
        'chunking',
        'history',
        'date_created',
        'related_store',
        'known_data_gaps',
        'verification',
    }
)


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
        '--variable',
        type=str,
        default=None,
        help='Verify the store holding this one variable. The config names the '
        'whole family of 14, which is what the checks that read across stores '
        'need; this names the one store being checked.',
    )
    parser.add_argument(
        '--phases',
        type=str,
        default=','.join(DEFAULT_PHASES),
        help=f'Comma separated subset of {",".join(PHASES)}. '
        'metadata needs --compare-with and is not run by default.',
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=42,
        help='Random boxes to compare, spread evenly over the variables. '
        'A box is one chunk of the store: a plane on the spatial layout, a '
        'lat/lon tile through the whole record on the temporal one.',
    )
    parser.add_argument(
        '--seed', type=int, default=20260910, help='Seed for the sample draw.'
    )
    parser.add_argument(
        '--identity-days',
        type=int,
        default=4,
        help='Random positions at which to test the component identities: '
        'days on the spatial layout, tiles on the temporal one.',
    )
    parser.add_argument(
        '--range-days',
        type=int,
        default=6,
        help='Random positions per variable for the bounds check.',
    )
    parser.add_argument(
        '--smallest',
        type=int,
        default=3,
        help='Smallest written chunks per variable to read back and compare '
        'against raw on the temporal layout, where a size floor cannot tell a '
        'sparse chunk from a truncated one.',
    )
    parser.add_argument(
        '--trace-absent',
        type=int,
        default=24,
        help='Budget, shared over the variables, for tracing absent tiles that '
        'another variable did write. Each one is read from raw across the whole '
        'record, which settles it rather than sampling it.',
    )
    parser.add_argument(
        '--compare-with',
        type=str,
        default=None,
        help='Config of a sibling store to compare metadata against. Required '
        'by the metadata phase and ignored by every other one.',
    )
    parser.add_argument(
        '--mask-days',
        type=int,
        default=12,
        help='Raw planes per variable used to justify the absent chunks on the '
        'temporal layout; unused on the spatial one, where every absent chunk '
        'is traced to its own raw plane.',
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


def sibling_session(settings, variable):
    """Open another variable's store of the same family, read only.

    The family shares one filename template, so a sibling's path is the same
    template rendered with a different variable -- there is no second naming
    convention to keep in step.

    Args:
        settings (dict): The loaded configuration.
        variable (str): The variable whose store to open.

    Returns:
        icechunk.Session: A read-only session, or None if that store is absent.
    """
    path = store_path({**settings, 'variable': variable})
    if not os.path.exists(path):
        return None
    repository = icechunk.Repository.open(icechunk.local_filesystem_storage(path))
    return repository.readonly_session(branch=BRANCH)


def ensure_indexed(index, settings, variables):
    """Extend the raw index with any of these variables it does not hold.

    The index is built for the store under test, which on a per-variable store
    is one variable. A check that falls back to raw for other variables -- the
    identities do, where a sum does not close -- pays for those files only if it
    actually needs them.

    Args:
        index (dict): The raw index, modified in place.
        settings (dict): The loaded configuration.
        variables (list): Variable names the caller is about to read.

    Returns:
        dict: The index.
    """
    missing = [name for name in variables if name not in index]
    if missing:
        index.update(build_raw_index(settings, missing))
    return index


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


def store_layout(chunks, sizes):
    """Which access pattern the store's chunking is built for.

    Args:
        chunks (dict): Resolved chunk sizes.
        sizes (Mapping): Length of each dimension, e.g. ``dataset.sizes``.

    Returns:
        str: 'temporal' if one chunk spans the whole record, else 'spatial'.
    """
    return 'temporal' if chunks['time'] >= sizes['time'] else 'spatial'


def box_shape(layout, chunks, sizes):
    """The box that is one chunk of this store, as lengths per dimension.

    This is the only place the two layouts differ in what a cheap read is, and
    everything that reads data sizes itself from here. Reading the other
    layout's box would still be correct and would cost the whole store: a
    global plane out of a time-chunked store touches every chunk the variable
    has.

    Args:
        layout (str): From ``store_layout``.
        chunks (dict): Resolved chunk sizes.
        sizes (Mapping): Length of each dimension, e.g. ``dataset.sizes``.

    Returns:
        dict: Box length per dimension.
    """
    if layout == 'temporal':
        return {
            'time': sizes['time'],
            'lat': chunks['lat'],
            'lon': chunks['lon'],
        }
    return {'time': 1, 'lat': sizes['lat'], 'lon': sizes['lon']}


def read_raw_box(index, name, t0, t1, lat=slice(None), lon=slice(None)):
    """Read a (time, lat, lon) box from the raw files, fill sentinel intact.

    The box is assembled from however many yearly files its time range spans,
    so a box covering the whole record reads every file of the variable. Each
    file contributes one contiguous hyperslab, which is what keeps the read
    proportional to the box rather than to the source chunks around it.

    Args:
        index (dict): The raw index from ``build_raw_index``.
        name (str): Variable to read.
        t0 (int): First global timestep, inclusive.
        t1 (int): Last global timestep, exclusive.
        lat (slice): Latitude range to take.
        lon (slice): Longitude range to take.

    Returns:
        numpy.ndarray: The float32 box, still holding -999 where masked.
    """
    meta = index[name]
    pieces = []
    timestep = t0
    while timestep < t1:
        which = int(np.searchsorted(meta['offsets'], timestep, side='right') - 1)
        local = int(timestep - meta['offsets'][which])
        take = int(min(t1 - timestep, meta['lengths'][which] - local))
        dataset = netCDF4.Dataset(meta['files'][which])
        # masking and scaling off, so the comparison does not run through the
        # same CF decoding the pipeline used and cannot agree by construction
        dataset.set_auto_mask(False)
        dataset.set_auto_scale(False)
        pieces.append(
            np.asarray(dataset.variables[name][local:local + take, lat, lon])
            .astype(np.float32)
        )
        dataset.close()
        timestep += take
    return np.concatenate(pieces, axis=0)


def read_raw_plane(index, name, timestep):
    """Read one whole lat/lon plane from raw, fill sentinel intact.

    Args:
        index (dict): The raw index from ``build_raw_index``.
        name (str): Variable to read.
        timestep (int): Global timestep, as indexed in the store.

    Returns:
        numpy.ndarray: The float32 plane, still holding -999 where masked.
    """
    return read_raw_box(index, name, timestep, timestep + 1)[0]


def compare_box(stored, raw):
    """Compare a box read from the store against its raw counterpart.

    Args:
        stored (numpy.ndarray): The decoded box from the store.
        raw (numpy.ndarray): The raw box, fill sentinel intact.

    Returns:
        tuple: Whether the NaN pattern matches the fill pattern, whether every
            non-fill cell is bit-identical, and the count of valid cells.
    """
    is_fill = raw == RAW_FILL
    mask_ok = np.array_equal(is_fill, np.isnan(stored))
    values_ok = bool(np.all(stored[~is_fill] == raw[~is_fill]))
    return mask_ok, values_ok, int((~is_fill).sum())


def tile_grid(chunks, sizes):
    """The store's chunk grid over lat and lon.

    Args:
        chunks (dict): Resolved chunk sizes.
        sizes (Mapping): Length of each dimension, e.g. ``dataset.sizes``.

    Returns:
        tuple: Number of chunks along lat and along lon.
    """
    return (
        -(-sizes['lat'] // chunks['lat']),
        -(-sizes['lon'] // chunks['lon']),
    )


def tile_valid_counts(plane, chunks, grid):
    """Count the cells that are not fill in each tile of the chunk grid.

    Args:
        plane (numpy.ndarray): A raw plane, fill sentinel intact.
        chunks (dict): Resolved chunk sizes.
        grid (tuple): Number of chunks along lat and lon.

    Returns:
        numpy.ndarray: Integer (n_lat_tiles, n_lon_tiles) counts.
    """
    valid = plane != RAW_FILL
    return valid.reshape(grid[0], chunks['lat'], grid[1], chunks['lon']).sum(
        axis=(1, 3)
    )


def coarsen_to_tiles(plane, chunks, grid):
    """Reduce a full plane's valid-cell mask onto the store's chunk grid.

    Args:
        plane (numpy.ndarray): A raw plane, fill sentinel intact.
        chunks (dict): Resolved chunk sizes.
        grid (tuple): Number of chunks along lat and lon.

    Returns:
        numpy.ndarray: Boolean (n_lat_tiles, n_lon_tiles), True where the tile
            holds at least one cell that is not fill.
    """
    return tile_valid_counts(plane, chunks, grid) > 0


def land_strata(index, name, chunks, grid, timestep):
    """Split the chunk grid into tiles that are fully, partly and never covered.

    One plane from the middle of the record is enough to say where the land is:
    the valid footprint moves from day to day, but not between ocean and land.

    Args:
        index (dict): The raw index from ``build_raw_index``.
        name (str): Variable whose footprint to take.
        chunks (dict): Resolved chunk sizes.
        grid (tuple): Number of chunks along lat and lon.
        timestep (int): Which plane to read.

    Returns:
        tuple: Three (n, 2) arrays of tile indices -- fully covered, partly
            covered, and holding nothing at all on that day.
    """
    counts = tile_valid_counts(read_raw_plane(index, name, timestep), chunks, grid)
    cells = chunks['lat'] * chunks['lon']
    return (
        np.argwhere(counts == cells),
        np.argwhere((counts > 0) & (counts < cells)),
        np.argwhere(counts == 0),
    )


def grid_divides(chunks, sizes):
    """Whether the lat/lon grid is a whole number of chunks.

    A raw plane can only be coarsened onto the chunk grid when it is, so the
    checks that do that fall back to something weaker when it is not.

    Args:
        chunks (dict): Resolved chunk sizes.
        sizes (Mapping): Length of each dimension, e.g. ``dataset.sizes``.

    Returns:
        bool: True if lat and lon are both whole multiples of their chunk.
    """
    return not (sizes['lat'] % chunks['lat'] or sizes['lon'] % chunks['lon'])


def tile_box(tile, chunks, sizes):
    """The (lat, lon) slices of one tile of the chunk grid.

    Args:
        tile (tuple): Tile index along lat and lon.
        chunks (dict): Resolved chunk sizes.
        sizes (Mapping): Length of each dimension, e.g. ``dataset.sizes``.

    Returns:
        tuple: The latitude and longitude slices, clipped at the grid edge.
    """
    i, j = int(tile[0]), int(tile[1])
    return (
        slice(i * chunks['lat'], min((i + 1) * chunks['lat'], sizes['lat'])),
        slice(j * chunks['lon'], min((j + 1) * chunks['lon'], sizes['lon'])),
    )


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

    chunks = resolve_chunks(settings['chunks'], dataset.sizes)
    layout = store_layout(chunks, dataset.sizes)
    report.note(f'chunked for {layout} access, one chunk is {box_shape(layout, chunks, dataset.sizes)}')
    # in dimension order rather than config order, so a reordered config cannot
    # turn a real mismatch into a passing check
    expected = tuple(chunks[dimension] for dimension in ('time', 'lat', 'lon'))
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

    # on the append path the time coordinate is chunked one commit batch per
    # chunk, which is what kept every append landing on a chunk boundary; the
    # region path has no batches and writes it whole
    if layout == 'temporal':
        expected_time_chunk = n_time
        label = f'time coordinate is a single chunk ({n_time})'
    else:
        expected_time_chunk = commit_batch_size(chunks, settings)
        label = f'time coordinate chunked one commit batch ({expected_time_chunk}) per chunk'
    report.check(
        label,
        root['time'].chunks == (expected_time_chunk,),
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

    # the chunking attribute is the one piece of prose a reader is expected to
    # act on, and prose does not follow a config edit the way the arrays do, so
    # the shape it claims is checked against the shape that was written
    if 'chunking' in dataset.attrs:
        claimed = f'({", ".join(str(v) for v in expected)})'
        report.check(
            f'chunking attribute describes this store, {claimed}',
            claimed in dataset.attrs['chunking'],
            f'{dataset.attrs["chunking"][:80]}...',
        )

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


def sample_spots(dataset, index, variables, chunks, layout, rng, count):
    """Choose positions to read, shaped as the store's own chunk.

    A spot is a box without a variable: the same position is read for every
    variable an identity or a bounds check needs, so they line up.

    On the temporal layout the draw is taken from the tiles that hold land
    rather than uniformly from the grid. Most of a 20x20 tiling of this grid is
    ocean, and an all-fill spot has no identity to close and no bound to check,
    so a uniform draw quietly spends the whole budget on nothing -- the phases
    would report success having tested nothing at all. Covered and coastal
    tiles alternate, because a component sum closing over open land and one
    closing along a coastline are not the same test.

    Args:
        dataset (xarray.Dataset): The decoded store.
        index (dict): The raw index from ``build_raw_index``.
        variables (list): Variable names, the first of which supplies the
            footprint the draw is stratified on.
        chunks (dict): Resolved chunk sizes.
        layout (str): From ``store_layout``.
        rng (numpy.random.Generator): Source of the draw.
        count (int): How many spots to draw.

    Returns:
        list: (t0, t1, lat_slice, lon_slice) boxes.
    """
    sizes = dataset.sizes
    if layout == 'spatial':
        days = sorted(set(int(x) for x in rng.integers(sizes['time'], size=count)))
        return [(t, t + 1, slice(None), slice(None)) for t in days]

    grid = tile_grid(chunks, sizes)
    tiles = []
    if grid_divides(chunks, sizes):
        covered, coastal, _ = land_strata(
            index, variables[0], chunks, grid, sizes['time'] // 2
        )
        drawn = []
        half = -(-count // 2)
        for stratum in (covered, coastal):
            if not len(stratum):
                continue
            picks = rng.choice(
                len(stratum), size=min(len(stratum), half), replace=False
            )
            drawn.append([tuple(int(v) for v in stratum[k]) for k in picks])
        # alternate the strata so a short one does not hand the whole draw to
        # the other
        while drawn and len(tiles) < count:
            for group in list(drawn):
                if not group:
                    drawn.remove(group)
                    continue
                tiles.append(group.pop())
                if len(tiles) >= count:
                    break
    while len(tiles) < count:
        tiles.append((int(rng.integers(grid[0])), int(rng.integers(grid[1]))))

    spots = []
    for tile in tiles:
        lat, lon = tile_box(tile, chunks, sizes)
        spots.append((0, sizes['time'], lat, lon))
    return spots


def describe_box(dataset, t0, t1, lat, lon):
    """A short label for a box, in whichever dimension it is narrow.

    Args:
        dataset (xarray.Dataset): The decoded store.
        t0 (int): First timestep, inclusive.
        t1 (int): Last timestep, exclusive.
        lat (slice): Latitude range.
        lon (slice): Longitude range.

    Returns:
        str: e.g. 't=8200 (2002-05-27)' or 'lat[600:620) lon[1200:1220)'.
    """
    if t1 - t0 == 1:
        return f't={t0} ({str(dataset["time"].values[t0])[:10]})'
    return (
        f'lat[{lat.start}:{lat.stop}) lon[{lon.start}:{lon.stop}) '
        f'x {t1 - t0} steps'
    )


def sample_boxes(dataset, index, variables, args, chunks, layout, settings):
    """Choose which boxes to compare against raw, stratified over the variables.

    On the spatial layout a box is a plane, so the draw adds the file seams and
    the commit batch boundaries deliberately: those are where an append or a
    resume would go wrong and random draws rarely land on them.

    On the temporal layout a box spans the whole record, so every box already
    crosses every file seam, every commit boundary and every gap day -- there is
    nothing left to add along time. What a draw can miss instead is *where*, so
    the tiles are stratified by how much land they hold, taken from one raw
    plane: a tile fully covered by data, a coastal tile partly covered, and an
    empty one exercise three different things, and the four corners are added
    because a transposed or rolled axis shows up there first.

    Args:
        dataset (xarray.Dataset): The decoded store.
        index (dict): The raw index.
        variables (list): Variable names.
        args (argparse.Namespace): Parsed arguments.
        chunks (dict): Resolved chunk sizes.
        layout (str): From ``store_layout``.
        settings (dict): The loaded configuration, for the commit batch size.

    Returns:
        list: (name, t0, t1, lat_slice, lon_slice) boxes to compare.
    """
    rng = np.random.default_rng(args.seed)
    sizes = dataset.sizes
    n_time = sizes['time']
    per_variable = max(1, args.samples // len(variables))

    if layout == 'spatial':
        picks = [
            (name, int(rng.integers(n_time)))
            for name in variables
            for _ in range(per_variable)
        ]
        # both sides of every yearly file seam, rotating the variable so the
        # seams spread over the variables too
        offsets = index[variables[0]]['offsets']
        for i, seam in enumerate(offsets[1:-1]):
            name = variables[i % len(variables)]
            picks.append((name, int(seam) - 1))
            picks.append((name, int(seam)))
        # the first and last timestep, and either side of each commit boundary
        batch = commit_batch_size(chunks, settings)
        boundaries = {0, n_time - 1, n_time - 2}
        for edge in range(batch, n_time, batch):
            boundaries.update({edge - 1, edge, edge + 1})
        for timestep in sorted(boundaries):
            if 0 <= timestep < n_time:
                picks.append((variables[int(rng.integers(len(variables)))], timestep))
        return [(name, t, t + 1, slice(None), slice(None)) for name, t in picks]

    grid = tile_grid(chunks, sizes)
    divides = grid_divides(chunks, sizes)
    boxes = []
    corners = [(0, 0), (0, grid[1] - 1), (grid[0] - 1, 0), (grid[0] - 1, grid[1] - 1)]
    for position, name in enumerate(variables):
        tiles = []
        if divides:
            # one plane from the middle of the record says which tiles hold land
            for stratum in land_strata(index, name, chunks, grid, n_time // 2):
                if len(stratum):
                    tiles.append(tuple(stratum[int(rng.integers(len(stratum)))]))
        while len(tiles) < per_variable:
            tiles.append((int(rng.integers(grid[0])), int(rng.integers(grid[1]))))
        # the corners rotate through the variables rather than being repeated
        tiles.append(corners[position % len(corners)])

        for tile in tiles:
            lat, lon = tile_box(tile, chunks, sizes)
            boxes.append((name, 0, n_time, lat, lon))
    return boxes


def check_samples(report, dataset, index, variables, args, chunks, layout, settings):
    """Compare boxes against raw, cell by cell.

    Args:
        report (Report): Where to record outcomes.
        dataset (xarray.Dataset): The decoded store.
        index (dict): The raw index.
        variables (list): Variable names.
        args (argparse.Namespace): Parsed arguments.
        chunks (dict): Resolved chunk sizes.
        layout (str): From ``store_layout``.
        settings (dict): The loaded configuration.
    """
    boxes = sample_boxes(dataset, index, variables, args, chunks, layout, settings)
    dates = dataset['time'].values
    n_cells = sum(
        (t1 - t0)
        * len(range(*lat.indices(dataset.sizes['lat'])))
        * len(range(*lon.indices(dataset.sizes['lon'])))
        for _, t0, t1, lat, lon in boxes
    )
    LOG.info(
        f'comparing {len(boxes)} boxes against raw, {n_cells / 1e6:.1f}M cells'
    )

    n_failed = 0
    for name, t0, t1, lat, lon in boxes:
        raw = read_raw_box(index, name, t0, t1, lat, lon)
        stored = dataset[name].isel(time=slice(t0, t1), lat=lat, lon=lon).values
        mask_ok, values_ok, _ = compare_box(stored, raw)

        # the store's own time values must be the ones the raw files carry at
        # those indices, or the box is right but landed on the wrong days
        stored_days = (
            (dates[t0:t1] - np.datetime64('1900-01-01')) / np.timedelta64(1, 'D')
        ).astype(np.int64)
        days_ok = np.array_equal(stored_days, index[name]['time'][t0:t1])

        if not (mask_ok and values_ok and days_ok):
            n_failed += 1
            report.check(
                f'{name:9s} {describe_box(dataset, t0, t1, lat, lon)} matches raw',
                False,
                f'mask={mask_ok} values={values_ok} days={days_ok}',
            )

    coverage = (
        f'all {len(variables)} variables and all '
        f'{len(index[variables[0]]["files"]) - 1} file seams'
        if layout == 'spatial'
        else f'all {len(variables)} variables over the whole record, so every '
        f'file seam and every commit boundary'
    )
    report.check(
        f'all {len(boxes)} sampled boxes bit-identical to raw, on the right days',
        n_failed == 0,
        f'{len(boxes) - n_failed} clean, {n_cells / 1e6:.1f}M cells, covering {coverage}',
    )


async def sibling_written_grids(settings, variables, grid):
    """Which tiles each sibling variable's store actually wrote.

    Read off each store's manifest, so a 14-store union costs metadata reads
    and no data I/O at all.

    Args:
        settings (dict): The loaded configuration.
        variables (list): Sibling variable names to look up.
        grid (tuple): Number of chunks along lat and lon.

    Returns:
        dict: Variable name -> boolean tile grid, skipping absent stores.
    """
    grids = {}
    for name in variables:
        session = sibling_session(settings, name)
        if session is None:
            continue
        written = np.zeros(grid, dtype=bool)
        async for coordinate in session.chunk_coordinates(f'/{name}'):
            written[coordinate[1], coordinate[2]] = True
        grids[name] = written
    return grids


def trace_absent_tiles(report, dataset, index, written, chunks, args, only=None):
    """Settle the absent tiles that another variable did write.

    The sweep justifies an absent tile against a handful of sampled raw planes,
    which cannot see a tile that holds data on no sampled day. The variables
    check each other instead: they share a land mask closely enough that a tile
    one variable wrote and another did not is the only interesting kind of hole,
    and there are few enough of them to settle outright by reading the whole
    record from raw. That is a proof rather than a sample, so it is worth its
    cost -- and it does have a cost, one whole-record read per tile, which is
    why the budget is shared over the variables rather than spent per variable.

    Args:
        report (Report): Where to record outcomes.
        dataset (xarray.Dataset): The decoded store.
        index (dict): The raw index.
        written (dict): Variable name -> boolean tile grid of written chunks.
        chunks (dict): Resolved chunk sizes.
        args (argparse.Namespace): Parsed arguments.
        only (list): Variables to trace, when the others are siblings that are
            only here to supply the union. Default is all of them.
    """
    sizes = dataset.sizes
    union = np.zeros(next(iter(written.values())).shape, dtype=bool)
    for tiles in written.values():
        union |= tiles

    suspect = {
        name: np.argwhere(union & ~tiles) for name, tiles in written.items()
    }
    pending = {
        name: tiles
        for name, tiles in suspect.items()
        if len(tiles) and (only is None or name in only)
    }
    if not pending:
        report.note(
            f'no absent tile needs tracing: every tile written by any of the '
            f'{len(written)} variables was written here too'
        )
        return

    budget = max(1, args.trace_absent // len(pending))
    rng = np.random.default_rng(args.seed)
    for name, tiles in pending.items():
        take = tiles
        if len(tiles) > budget:
            take = tiles[rng.choice(len(tiles), size=budget, replace=False)]
        lost = []
        for tile in take:
            lat, lon = tile_box(tile, chunks, sizes)
            raw = read_raw_box(index, name, 0, sizes['time'], lat, lon)
            if not bool(np.all(raw == RAW_FILL)):
                lost.append((int(tile[0]), int(tile[1])))
        report.check(
            f'{name:9s} absent tiles that another variable wrote are all-fill '
            f'in raw through the whole record',
            not lost,
            f'{len(take)} of {len(tiles)} such tiles read end to end'
            + (f'; DATA LOSS at {lost[:5]}' if lost else ''),
        )


async def check_sweep(
    report, session, dataset, index, variables, chunks, layout, args, settings, family
):
    """Check every chunk in the store for presence, placement and size.

    Runs off the manifest rather than by reading data, so it covers the whole
    store in seconds. An absent chunk is not assumed to be a defect: zarr does
    not write a chunk whose cells are all equal to the fill value, so a chunk
    holding nothing but fill correctly leaves a hole that reads back as NaN.

    What justifies a hole depends on the layout, because a hole means something
    different in each. On the spatial layout a chunk is one day, so a hole is a
    day and every one of them is traced to raw. On the temporal layout a chunk
    is a tile through the whole record, so a hole claims the tile never held
    data in 46 years -- which is ocean, and which no affordable read can confirm
    exhaustively. It is checked in the one direction that matters instead:
    every tile holding data on any sampled day must have been written. That
    catches real loss, and it does not mistake an unsampled day for one.

    Args:
        report (Report): Where to record outcomes.
        session (icechunk.Session): Read-only session on the store.
        dataset (xarray.Dataset): The decoded store.
        index (dict): The raw index.
        variables (list): Variable names.
        chunks (dict): Resolved chunk sizes.
        layout (str): From ``store_layout``.
        args (argparse.Namespace): Parsed arguments.
        settings (dict): The loaded configuration.
        family (list): Every variable of the family, which on a per-variable
            store is where the absent-tile trace finds something to compare
            against.
    """
    sizes = dataset.sizes
    n_time = sizes['time']
    dates = dataset['time'].values
    grid = tile_grid(chunks, sizes)
    n_time_chunks = -(-n_time // chunks['time'])
    n_grid = n_time_chunks * grid[0] * grid[1]
    floor = DEGENERATE_BYTES[layout]
    total_bytes = 0
    written_grids = {}

    mask_days = sorted(
        set(int(x) for x in np.linspace(0, n_time - 1, args.mask_days).round())
    )
    if layout == 'temporal':
        report.note(
            f'justifying absent chunks against {len(mask_days)} raw planes per '
            f'variable: {", ".join(str(dates[t])[:10] for t in mask_days[:4])}, ...'
        )

    for name in variables:
        coordinates = [c async for c in session.chunk_coordinates(f'/{name}')]
        report.check(
            f'{name:9s} every chunk inside the {n_time_chunks}x{grid[0]}x{grid[1]} '
            f'grid, none duplicated',
            len(coordinates) == len(set(coordinates))
            and all(
                0 <= c[0] < n_time_chunks and 0 <= c[1] < grid[0] and 0 <= c[2] < grid[1]
                for c in coordinates
            ),
            f'{len(coordinates)} of {n_grid} grid positions written',
        )

        if not coordinates:
            report.check(
                f'{name:9s} has at least one written chunk',
                False,
                'nothing in the manifest for this variable',
            )
            continue

        chunk_sizes = np.asarray(
            await asyncio.gather(
                *(
                    session.store.getsize(f'{name}/c/{t}/{i}/{j}')
                    for t, i, j in coordinates
                )
            )
        )
        total_bytes += int(chunk_sizes.sum())
        degenerate = [coordinates[k] for k in np.flatnonzero(chunk_sizes <= floor)]
        report.check(
            f'{name:9s} no written chunk is degenerate or truncated '
            f'(> {floor / 1024:g} KiB)',
            not degenerate,
            f'compressed min={chunk_sizes.min() / 1024**2:.3f} '
            f'median={np.median(chunk_sizes) / 1024**2:.3f} '
            f'max={chunk_sizes.max() / 1024**2:.3f} MiB'
            + (f', degenerate at {degenerate[:5]}' if degenerate else ''),
        )

        if layout == 'spatial':
            # a chunk is a day, so a hole is a day, and every one is traceable
            present = {c[0] for c in coordinates}
            missing = sorted(set(range(n_time)) - present)
            if not missing:
                continue
            real_loss = [
                timestep
                for timestep in missing
                if not bool(np.all(read_raw_plane(index, name, timestep) == RAW_FILL))
            ]
            report.check(
                f'{name:9s} every absent chunk is an all-fill raw plane',
                not real_loss,
                f'{len(missing)} absent, raw all-fill at every one'
                if not real_loss
                else f'DATA LOSS at {real_loss[:5]}',
            )
            report.note(
                f'{name} has no chunk on {len(missing)} days because raw is '
                f'entirely fill there: '
                f'{", ".join(str(dates[t])[:10] for t in missing)}'
            )
            continue

        if not grid_divides(chunks, sizes):
            report.note(
                f'{name} absent chunks not checked: the grid is not a whole '
                f'number of chunks, so a raw plane cannot be coarsened onto it'
            )
            continue
        ever = np.zeros(grid, dtype=bool)
        for timestep in mask_days:
            ever |= coarsen_to_tiles(
                read_raw_plane(index, name, timestep), chunks, grid
            )
        written = np.zeros(grid, dtype=bool)
        for _, i, j in coordinates:
            written[i, j] = True
        written_grids[name] = written
        lost = np.argwhere(ever & ~written)
        report.check(
            f'{name:9s} every tile holding data on a sampled day was written',
            not len(lost),
            f'{int(ever.sum())} tiles hold data on {len(mask_days)} sampled days, '
            f'{int(written.sum())} written'
            + (
                ''
                if not len(lost)
                else f'; DATA LOSS at {[tuple(int(v) for v in x) for x in lost[:5]]}'
            ),
        )
        # a floor cannot separate a sparse tile from a truncated one here, so
        # the smallest chunks are read back and compared instead. If a tile is
        # small because its data is sparse this passes; if it is small because
        # it was cut short, nothing else in the sweep would notice.
        order = [int(k) for k in np.argsort(chunk_sizes)[: max(0, args.smallest)]]
        damaged, decoded = [], []
        for k in order:
            lat, lon = tile_box(coordinates[k][1:], chunks, sizes)
            raw = read_raw_box(index, name, 0, n_time, lat, lon)
            stored = dataset[name].isel(
                time=slice(0, n_time), lat=lat, lon=lon
            ).values
            mask_ok, values_ok, valid_cells = compare_box(stored, raw)
            decoded.append((int(chunk_sizes[k]), valid_cells))
            if not (mask_ok and values_ok):
                damaged.append(coordinates[k])
        if decoded:
            report.check(
                f'{name:9s} the {len(order)} smallest written chunks decode '
                f'bit-identical to raw',
                not damaged,
                f'smallest {decoded[0][0]} B holds {decoded[0][1]} valid cells '
                f'of {n_time * chunks["lat"] * chunks["lon"]}'
                + (f'; DAMAGED at {damaged}' if damaged else ''),
            )
        else:
            report.note(f'{name} no chunk read back: --smallest is {args.smallest}')

        report.note(
            f'{name}: {n_grid - int(written.sum())} of {n_grid} tiles absent, '
            f'all fill through the whole record; '
            f'{int((written & ~ever).sum())} written tiles held no data on the '
            f'sampled days, which is legitimate -- one valid day anywhere in the '
            f'record is enough to write a tile'
        )

    if written_grids:
        # one variable per store leaves nothing here to compare against, so the
        # union is taken from the sibling stores' manifests instead. It is the
        # same check either way: a tile some other variable wrote and this one
        # did not is the only kind of hole worth reading raw to settle
        variable = settings.get('variable')
        if variable is not None:
            siblings = await sibling_written_grids(
                settings, [name for name in family if name != variable], grid
            )
            report.note(
                f'absent-tile union taken from {len(siblings)} sibling stores'
            )
            trace_absent_tiles(
                report,
                dataset,
                index,
                {**written_grids, **siblings},
                chunks,
                args,
                only=list(written_grids),
            )
        elif len(written_grids) > 1:
            trace_absent_tiles(report, dataset, index, written_grids, chunks, args)

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


def identity_sources(report, dataset, settings, total_name, components):
    """Where to read each side of an identity from.

    On an all-variable store every side is in the store under test. On a
    per-variable store the components live in sibling stores, and only the
    store holding the *total* runs the check -- the component stores would each
    repeat the identical comparison.

    Args:
        report (Report): Where to record outcomes.
        dataset (xarray.Dataset): The store under test.
        settings (dict): The loaded configuration.
        total_name (str): Left-hand side of the identity.
        components (tuple): Right-hand side.

    Returns:
        dict: Variable name -> the dataset holding it, or None if this store
            should not run this identity or a store is missing.
    """
    variable = settings.get('variable')
    if variable is not None and total_name != variable:
        return None

    sources, missing = {}, []
    for name in (total_name, *components):
        if name in dataset.data_vars:
            sources[name] = dataset
            continue
        if variable is None:
            # an all-variable store simply does not hold this identity
            return None
        session = sibling_session(settings, name)
        if session is None:
            missing.append(name)
        else:
            sources[name] = xr.open_zarr(session.store, consolidated=False)
    if missing:
        report.note(
            f'{total_name} identity not checked: no store found for {missing}'
        )
        return None
    return sources


def check_identities(report, dataset, index, variables, args, chunks, layout, settings):
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
        variables (list): Variable names.
        args (argparse.Namespace): Parsed arguments.
        chunks (dict): Resolved chunk sizes.
        layout (str): From ``store_layout``.
        settings (dict): The loaded configuration, for finding sibling stores.
    """
    rng = np.random.default_rng(args.seed)
    spots = sample_spots(
        dataset, index, variables, chunks, layout, rng, args.identity_days
    )

    for total_name, components in IDENTITIES:
        sources = identity_sources(report, dataset, settings, total_name, components)
        if sources is None:
            continue
        if sources[total_name] is not dataset:
            report.note(f'{total_name} identity read across {len(sources)} stores')
        tested = 0
        for t0, t1, lat, lon in spots:
            select = dict(time=slice(t0, t1), lat=lat, lon=lon)
            total = sources[total_name][total_name].isel(**select).values
            if not np.isfinite(total).any():
                # nothing but fill here, so there is no identity to close. The
                # count below is what keeps a draw that lands entirely on ocean
                # from reporting success having tested nothing
                report.note(
                    f'{total_name} is all fill at '
                    f'{describe_box(dataset, t0, t1, lat, lon)}, nothing to close'
                )
                continue
            tested += 1
            summed = sum(
                sources[name][name].isel(**select).values for name in components
            )
            residual = np.abs(total - summed)
            valid = np.isfinite(total) & np.isfinite(summed)
            worst = float(np.max(residual[valid])) if valid.any() else 0.0
            where = describe_box(dataset, t0, t1, lat, lon)
            label = f'{total_name} == {" + ".join(components)} at {where}'

            if worst < 1e-4:
                report.check(
                    f'{label} closes to float32 rounding',
                    True,
                    f'max|residual|={worst:.3e}',
                )
                continue

            # recompute the identity from raw, decoded the same way, and see
            # whether the store's residual field is the raw one exactly
            raw_boxes = {}
            # only now does raw for the other side matter, so only now is it
            # indexed -- on a per-variable store that is 6 more variables' files
            ensure_indexed(index, settings, (total_name, *components))
            for name in (total_name, *components):
                box = read_raw_box(index, name, t0, t1, lat, lon)
                box[box == RAW_FILL] = np.nan
                raw_boxes[name] = box
            raw_residual = np.abs(
                raw_boxes[total_name] - sum(raw_boxes[n] for n in components)
            )
            same = np.array_equal(
                np.nan_to_num(residual, nan=-1.0),
                np.nan_to_num(raw_residual, nan=-1.0),
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
                f'({100 * n_over / max(1, int(valid.sum())):.4f}%); '
                f'the store matches raw'
            )

        report.check(
            f'{total_name} identity tested somewhere that holds data',
            tested > 0,
            f'{tested} of {len(spots)} sampled positions held data',
        )


def check_ranges(report, dataset, index, variables, args, chunks, layout):
    """Check physical bounds, and report how the valid footprint moves.

    The bounds are asserted. The valid-cell footprint is only reported: in
    GLEAM v4.3a it genuinely moves from day to day for most variables, and the
    box comparisons already prove the NaN pattern matches raw exactly, so
    asserting a static mask would report upstream behaviour as a defect.

    Args:
        report (Report): Where to record outcomes.
        dataset (xarray.Dataset): The decoded store.
        index (dict): The raw index.
        variables (list): Variable names.
        args (argparse.Namespace): Parsed arguments.
        chunks (dict): Resolved chunk sizes.
        layout (str): From ``store_layout``.
    """
    rng = np.random.default_rng(args.seed)
    spots = sample_spots(
        dataset, index, variables, chunks, layout, rng, args.range_days
    )

    for name in variables:
        units = dataset[name].attrs.get('units')
        bounds = BOUNDS_BY_UNITS.get(units)
        counts, cells, lows, highs = [], [], [], []
        for t0, t1, lat, lon in spots:
            values = dataset[name].isel(time=slice(t0, t1), lat=lat, lon=lon).values
            finite = np.isfinite(values)
            counts.append(int(finite.sum()))
            cells.append(int(values.size))
            if finite.any():
                lows.append(float(np.min(values[finite])))
                highs.append(float(np.max(values[finite])))

        # a bound holds trivially over cells that are all NaN, so whether the
        # draw found any data is a finding in its own right rather than a
        # precondition to assume
        report.check(
            f'{name:9s} at least one sampled box holds data',
            bool(lows),
            f'{sum(1 for c in counts if c)} of {len(spots)} boxes hold data',
        )
        if bounds is None:
            report.note(f'{name} has units {units!r} with no bounds to check')
            continue
        if not lows:
            continue
        low, high = bounds
        observed = (min(lows), max(highs))
        report.check(
            f'{name:9s} within [{low:g}, {high:g}] {units}',
            observed[0] >= low and observed[1] <= high,
            f'observed [{observed[0]:.4g}, {observed[1]:.4g}] over {len(spots)} boxes',
        )

        if layout == 'temporal':
            # a box already spans the record, so the footprint's movement is
            # inside it and check_samples has already compared that NaN pattern
            # against raw cell by cell; there is nothing a re-read would add
            report.note(
                f'{name} valid cells per sampled box: '
                f'{min(counts)}..{max(counts)} of '
                f'{max(cells)} ({len(spots)} boxes)'
            )
            continue

        # a moving footprint is upstream; confirm the store follows raw on the
        # day it is widest apart rather than asserting it stays put
        if len(set(counts)) > 1:
            t0 = spots[int(np.argmin(counts))][0]
            plane_raw = read_raw_plane(index, name, t0)
            stored = dataset[name].isel(time=t0).values
            mask_ok, _, n_valid = compare_box(stored, plane_raw)
            report.check(
                f'{name:9s} moving valid footprint follows raw exactly',
                mask_ok,
                f'valid cells {min(counts)}..{max(counts)} over {len(spots)} days, '
                f'{n_valid} at {str(dataset["time"].values[t0])[:10]}',
            )
        else:
            report.note(f'{name} valid footprint is static at {counts[0]} cells')


def check_metadata(report, dataset, other_settings):
    """Compare a store's metadata against a sibling store's.

    Two stores built from the same raw files hold the same data and must
    describe it the same way. Almost everything they carry is either taken
    straight from the source files or is shared prose from the config, so a
    difference is a defect -- the exceptions are the handful of attributes that
    describe the layout, the build or what has been proven about it, which are
    reported side by side instead of asserted. Metadata and three coordinate
    arrays, so it costs seconds however large the stores are.

    Args:
        report (Report): Where to record outcomes.
        dataset (xarray.Dataset): The decoded store being verified.
        other_settings (dict): The loaded configuration of the store to compare
            against.
    """
    path = store_path(other_settings)
    LOG.info(f'comparing metadata against {path}')
    repository = icechunk.Repository.open(icechunk.local_filesystem_storage(path))
    other = xr.open_zarr(
        repository.readonly_session(branch=BRANCH).store, consolidated=False
    )

    here, there = set(dataset.data_vars), set(other.data_vars)
    report.check(
        'both stores hold the same variables',
        here == there,
        f'{len(here)} here, {len(there)} there'
        + (f'; only here {sorted(here - there)}' if here - there else '')
        + (f'; only there {sorted(there - here)}' if there - here else ''),
    )

    mismatched = []
    for name in sorted(here & there):
        if dataset[name].dims != other[name].dims:
            mismatched.append(f'{name} dims')
        if dataset[name].dtype != other[name].dtype:
            mismatched.append(f'{name} dtype')
        differing = sorted(
            key
            for key in set(dataset[name].attrs) | set(other[name].attrs)
            if dataset[name].attrs.get(key) != other[name].attrs.get(key)
        )
        if differing:
            mismatched.append(f'{name} attrs {differing}')
    report.check(
        'every shared variable has identical dims, dtype and attributes',
        not mismatched,
        f'{len(here & there)} variables compared'
        + (f'; differing {mismatched[:5]}' if mismatched else ''),
    )

    for coordinate in ('time', 'lat', 'lon'):
        mine, theirs = dataset[coordinate].values, other[coordinate].values
        report.check(
            f'{coordinate:9s} coordinate identical in both stores',
            mine.shape == theirs.shape and np.array_equal(mine, theirs),
            f'{mine.size} values here, {theirs.size} there',
        )

    keys = (set(dataset.attrs) | set(other.attrs)) - LAYOUT_ATTRS
    differing = sorted(
        key for key in keys if dataset.attrs.get(key) != other.attrs.get(key)
    )
    report.check(
        'global attributes outside the layout-specific ones are identical',
        not differing,
        f'{len(keys)} compared, {len(LAYOUT_ATTRS)} exempt'
        + (f'; differing {differing}' if differing else ''),
    )
    for key in sorted(LAYOUT_ATTRS & (set(dataset.attrs) | set(other.attrs))):
        report.note(
            f'{key}: here {str(dataset.attrs.get(key, "<absent>")):.90} | '
            f'there {str(other.attrs.get(key, "<absent>")):.90}'
        )


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
    chunks = resolve_chunks(settings['chunks'], dataset.sizes)
    layout = store_layout(chunks, dataset.sizes)

    if layout == 'temporal':
        block_shape = resolve_block_shape(settings, chunks, dataset.sizes)
        units = len(
            iter_blocks(sorted(dataset.data_vars), dataset.sizes, block_shape)
        )
        # one commit per block, plus the skeleton, plus the repository's own
        # initial snapshot
        expected = units + 2
        noun = f'{units} block commits, the skeleton'
    else:
        batch = commit_batch_size(chunks, settings)
        units = -(-dataset.sizes['time'] // batch)
        # one commit per batch, plus the repository's own initial snapshot
        expected = units + 1
        noun = f'{units} batch commits'
    # a finalization commit -- provenance attributes, for instance -- legitimately
    # adds to this, so the build's commit count is a floor rather than an equality
    report.check(
        f'at least one reachable snapshot per commit ({noun}, and the initial one)',
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
        ValueError: If --phases names something that is not a phase, names the
            metadata phase without a store to compare against, or names a
            variable the config's family does not hold.
    """
    phases = [p.strip() for p in args.phases.split(',') if p.strip()]
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        raise ValueError(f'unknown phases {unknown}; choose from {list(PHASES)}')
    if 'metadata' in phases and not args.compare_with:
        raise ValueError('the metadata phase needs --compare-with <config>')

    report = Report()
    # the config names the family; --variable names the one store under test.
    # Both are needed: the checks that read across stores resolve siblings from
    # the family, and the store itself must hold exactly the one
    family = resolve_variables(settings)
    variables = family
    if args.variable:
        if args.variable not in family:
            raise ValueError(f'{args.variable!r} is not in the family {family}')
        settings['variable'] = args.variable
        variables = [args.variable]
    repository, session, dataset = open_store(settings)

    # the chunking decides what a cheap read is, so it is resolved once here and
    # every phase that reads data is handed it rather than assuming a layout
    chunks = resolve_chunks(settings['chunks'], dataset.sizes)
    layout = store_layout(chunks, dataset.sizes)
    LOG.info(f'phases: {phases}, chunks {chunks}, layout {layout!r}')

    # every phase but 'structure' and 'metadata' traces store values back to a
    # raw file
    index = (
        build_raw_index(settings, variables)
        if set(phases) - {'structure', 'metadata'}
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
        check_samples(
            report, dataset, index, variables, args, chunks, layout, settings
        )
    if 'sweep' in phases:
        LOG.info('--- sweep')
        asyncio.run(
            check_sweep(
                report,
                session,
                dataset,
                index,
                variables,
                chunks,
                layout,
                args,
                settings,
                family,
            )
        )
    if 'identities' in phases:
        LOG.info('--- identities')
        check_identities(
            report, dataset, index, variables, args, chunks, layout, settings
        )
    if 'ranges' in phases:
        LOG.info('--- ranges')
        check_ranges(report, dataset, index, variables, args, chunks, layout)
    if 'metadata' in phases:
        LOG.info('--- metadata')
        other = load_config(args.compare_with)
        if args.variable:
            # the sibling layout holds the same variable in its own store
            other['variable'] = args.variable
        check_metadata(report, dataset, other)

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
    # named after the store's own log file, so two stores' forensics do not
    # interleave in one file
    setup_logging(
        os.path.join(
            configuration['directories']['logs'],
            f'verify_{configuration["log_file"]}',
        )
    )
    sys.exit(main(configuration, arguments))
