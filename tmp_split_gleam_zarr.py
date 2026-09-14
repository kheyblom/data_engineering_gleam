"""One-off: split an all-variable GLEAM store into one icechunk store per variable.

**Temporary.** The pipeline is to build per-variable stores directly, and this
exists only to carry the two finished all-variable stores over without paying
for the raw netCDF decompression a second time -- that was ~97% of what they
cost to build. Delete it once the originals are gone.

It sits at the repo root rather than in a scratch directory for one reason: a
script's own directory is what lands on ``sys.path``, so ``from utils...`` only
resolves from here. It is deliberately untracked.

The data is read from the finished store and written through the pipeline's own
write path -- ``create_skeleton``/``write_by_region`` for the time-series
layout, ``write_dataset`` for the map layout -- so the new stores are written
exactly the way the originals were, by code that has already done it at this
scale twice.

Copying the stored chunk bytes directly would avoid decoding altogether and was
tried first. It does not work: the copy rate collapses as the destination grows
(202 chunks/s over the first 2000, 41 over the next, 18 over the next, whether
or not the writes are committed in batches), because every commit rewrites a
manifest that holds every chunk reference written so far. Going through the
normal write path costs decode and re-encode, and is the only one that holds a
steady rate.

Each variable's store is independent, so variables can be split in parallel
processes with no coordination -- there is no shared branch to contend over.

A store is built under ``<name>.partial`` and renamed only once it is finished,
so a walltime kill can never leave something a later run mistakes for a
complete store.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time

import xarray as xr

from utils.log_utils import setup_logging
from utils.path_utils import load_config, resolve_variables, store_path
from utils.zarr_utils import (
    BRANCH,
    build_encoding,
    check_resume,
    check_resume_region,
    commit_batch_size,
    committed_blocks,
    committed_timesteps,
    configure_runtime,
    create_skeleton,
    iter_blocks,
    open_existing_repository,
    open_repository,
    resolve_block_shape,
    resolve_chunks,
    resolve_write_strategy,
    write_by_region,
    write_dataset,
)

LOG = logging.getLogger(__name__)

COORDINATES = ('time', 'lat', 'lon')

# a store is built under this suffix and renamed once it is complete
PARTIAL_SUFFIX = '.partial'

# lat rows per region block. Far smaller than the build's 200: that was sized
# against the raw files' [12, 57, 113] chunking, where a narrow block
# decompresses the same source chunk once per tile it covers. Reading from a
# store chunked (16802, 20, 20) there is no amplification at all as long as the
# block is a whole number of chunks, so the block only has to be big enough to
# keep commits down, and a smaller one costs less memory.
DEFAULT_BLOCK_LAT = 100


def parse_args():
    """Parse the command line.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='split a store into one store per variable.')
    parser.add_argument(
        '--config', required=True, help='Config of the all-variable source store.'
    )
    parser.add_argument(
        '--sibling-config',
        required=True,
        help='Config of the other layout, to name in related_store.',
    )
    parser.add_argument(
        '--variables', default=None, help='Comma separated subset; default is all.'
    )
    parser.add_argument(
        '--block-lat',
        type=int,
        default=DEFAULT_BLOCK_LAT,
        help='Lat rows per region block on the time-series layout.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Override num_workers. One process per variable is the parallelism '
        'that matters here, so a fan-out job passes 1 and leaves the cpus to the '
        'other variables.',
    )
    parser.add_argument(
        '--dry-run', action='store_true', help='Report what would be written.'
    )
    return parser.parse_args()


def layout_of(settings, chunks, sizes):
    """Which access pattern the source store's chunking is built for.

    Args:
        settings (dict): The loaded configuration.
        chunks (dict): Resolved chunk sizes.
        sizes (Mapping): Length of each dimension.

    Returns:
        str: 'temporal' if one chunk spans the whole record, else 'spatial'.
    """
    return 'temporal' if chunks['time'] >= sizes['time'] else 'spatial'


def destination_path(settings, variable):
    """Where one variable's store goes: a directory per layout.

    The layout lives in the directory and the variable in the name, so the
    source store's own suffix -- 'spatial', 'temporal', or a fixture's -- names
    the directory and is what the variable replaces in the filename.

    Args:
        settings (dict): The loaded configuration of the source store.
        variable (str): The variable the store holds.

    Returns:
        str: Full path of the per-variable store.
    """
    suffix = settings['output_conventions']['suffix']
    source = store_path(settings)
    name = os.path.basename(source).replace(f'.{suffix}.zarr', f'.{variable}.zarr')
    return os.path.join(os.path.dirname(source), suffix, name)


def single_variable_attrs(attrs, variable, variable_attrs, layout, sibling, source):
    """Rewrite the all-variable store's attributes for a one-variable store.

    Args:
        attrs (dict): The source store's global attributes.
        variable (str): The variable this store holds.
        variable_attrs (dict): That variable's own attributes.
        layout (str): 'spatial' or 'temporal'.
        sibling (str): Name of the same variable's store in the other layout.
        source (str): Name of the store this was split from.

    Returns:
        dict: Attributes for the new store.
    """
    out = dict(attrs)
    # every GLEAM long_name ends '... from GLEAM 4.3a', which reads twice over
    # once it sits inside a title that already says which dataset this is
    long_name = variable_attrs.get('long_name', variable).split(' from GLEAM ')[0]
    units = variable_attrs.get('units', '')

    # a claim about a store is earned by verifying that store, and this one is
    # new. finalize_gleam_zarr.py --attrs adds it back once it has been
    out.pop('verification', None)

    chunked_for = 'time series' if layout == 'temporal' else 'maps and fields'
    out['title'] = (
        f'GLEAM v4.3a daily {long_name[0].lower() + long_name[1:]} ({variable}), '
        f'native 0.1 degree global grid, 1980-2025, chunked for {chunked_for}'
    )
    out['summary'] = (
        f'{long_name} ({variable}, {units}) from GLEAM v4.3a daily, on the native '
        f'0.1 degree global grid over 1980-2025. One variable per store: the other '
        f'13 GLEAM variables are in sibling stores beside this one, on the same '
        f'grid and the same time axis.'
    )
    out['related_store'] = (
        f'{sibling} -- the same variable on the same grid over the same period, '
        f'chunked for the opposite access pattern. Identical values; the two '
        f'differ only in chunking.'
    )
    out['history'] = (
        f'{attrs.get("history", "").strip()} '
        f'{datetime.date.today().isoformat()}: {variable} written into its own '
        f'store from {source}, which was built from the raw files and verified '
        f'against them.'
    ).strip()

    # the gap attribute describes E, and half of it describes this layout. A
    # store must not carry a note about data it does not hold
    gaps = []
    if variable == 'E':
        gaps.append(
            'This variable is entirely fill (NaN) on 25 days -- five 5-day blocks, '
            '1982-02-05..02-09 and 1992-01-01..01-05, 03-21..03-25, 05-20..05-24 '
            'and 10-02..10-06 -- while every other GLEAM variable, including E\'s '
            'own components, has data on those days. This is a gap in upstream '
            'GLEAM v4.3a, not a conversion defect.'
        )
        if layout == 'spatial':
            gaps.append(
                'Zarr does not write a chunk whose cells are all fill, so this '
                'store holds 16777 chunks against 16802 timesteps and that is '
                'correct.'
            )
    if layout == 'temporal':
        gaps.append(
            'Chunks absent from this store are the 2x2 degree tiles that are fill '
            'across the entire record, which is ocean; zarr does not write a chunk '
            'whose cells are all fill. More than half the chunk grid is absent for '
            'this reason and that is correct.'
        )
    if gaps:
        out['known_data_gaps'] = ' '.join(gaps)
    else:
        out.pop('known_data_gaps', None)
    return out


def split_one(settings, source, variable, layout, sibling_name, block_lat):
    """Write one variable's store.

    Args:
        settings (dict): The loaded configuration of the source store.
        source (xarray.Dataset): The open all-variable store.
        variable (str): Variable to extract.
        layout (str): 'spatial' or 'temporal'.
        sibling_name (str): Store name to quote in related_store.
        block_lat (int): Lat rows per region block.

    Returns:
        str: The path written.
    """
    destination = destination_path(settings, variable)
    partial = destination + PARTIAL_SUFFIX
    resuming = os.path.exists(partial)
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    dataset = source[[variable]]
    dataset.attrs = single_variable_attrs(
        source.attrs,
        variable,
        source[variable].attrs,
        layout,
        sibling_name,
        os.path.basename(store_path(settings)),
    )
    chunks = resolve_chunks(settings['chunks'], dataset.sizes)
    strategy = resolve_write_strategy(settings, chunks, dataset.sizes)
    repository = open_repository(partial)
    started = time.time()

    # a partial store is resumed rather than discarded. The '.partial' name is
    # what keeps it from being mistaken for a finished store, so resuming into
    # it is safe, and at ~4 h per variable a walltime kill that threw the work
    # away would make the job unschedulable inside the 6 h develop cap
    if strategy == 'region':
        encoding = build_encoding(dataset, chunks, dataset.sizes['time'])
        block_shape = resolve_block_shape(
            {**settings, 'block_shape': {'lat': block_lat, 'lon': -1}},
            chunks,
            dataset.sizes,
        )
        blocks = iter_blocks([variable], dataset.sizes, block_shape)
        done = frozenset()
        if resuming:
            check_resume_region(repository, dataset, chunks)
            done = committed_blocks(repository)
            LOG.info(f'{variable}: resuming, {len(done)} of {len(blocks)} blocks done')
        else:
            create_skeleton(repository, dataset, encoding)
        LOG.info(f'{variable}: {len(blocks)} blocks of {block_shape}')
        write_by_region(repository, dataset, blocks, done)
    else:
        batch = commit_batch_size(chunks, settings)
        encoding = build_encoding(dataset, chunks, batch)
        start = 0
        if resuming:
            start = committed_timesteps(repository)
            check_resume(repository, dataset, start)
            LOG.info(
                f'{variable}: resuming, {start} of {dataset.sizes["time"]} timesteps done'
            )
        LOG.info(f'{variable}: appending in batches of {batch} timesteps from {start}')
        write_dataset(repository, dataset, encoding, batch, start)

    os.rename(partial, destination)
    LOG.info(
        f'{variable}: wrote {os.path.basename(destination)} in '
        f'{(time.time() - started) / 60:.1f} min'
    )
    return destination


def main():
    """Split every requested variable out of the configured store.

    Returns:
        int: 0 on success.
    """
    args = parse_args()
    settings = load_config(args.config)
    sibling_settings = load_config(args.sibling_config)
    if args.workers is not None:
        settings['num_workers'] = args.workers
    configure_runtime(settings)

    path = store_path(settings)
    repository = open_existing_repository(path)
    source = xr.open_zarr(repository.readonly_session(branch=BRANCH).store, consolidated=False)
    chunks = resolve_chunks(settings['chunks'], source.sizes)
    layout = layout_of(settings, chunks, source.sizes)

    present = [name for name in source.data_vars if name not in COORDINATES]
    wanted = [v.strip() for v in args.variables.split(',')] if args.variables else present
    missing = [v for v in wanted if v not in present]
    if missing:
        raise SystemExit(f'not in the source store: {missing}')

    LOG.info(f'splitting {path}')
    LOG.info(f'{layout} layout, chunks {chunks}, {len(wanted)} variables: {wanted}')

    for variable in wanted:
        destination = destination_path(settings, variable)
        if os.path.exists(destination):
            LOG.info(f'{variable}: already written, skipping {destination}')
            continue
        if args.dry_run:
            LOG.info(f'{variable}: would write {destination}')
            continue
        sibling = destination_path(sibling_settings, variable)
        # the layout lives in the directory, so both stores share a filename:
        # without the directory this attribute would name the store it is on
        sibling_name = os.path.join(*sibling.split(os.sep)[-2:])
        split_one(settings, source, variable, layout, sibling_name, args.block_lat)
    return 0


if __name__ == '__main__':
    arguments = parse_args()
    configuration = load_config(arguments.config)
    setup_logging(
        os.path.join(
            configuration['directories']['logs'], f'split_{configuration["log_file"]}'
        )
    )
    sys.exit(main())
