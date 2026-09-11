"""Build an icechunk backed zarr store from the raw GLEAM netCDF files.

The raw tree written by the download step holds one directory per variable, each
with one netCDF file per year:
``<download>/<version>/raw/<temporal_resolution>/<variable>/<file>.nc``, with the
version written as ``v_4_3_a`` rather than ``v4.3a``. This script opens every
configured variable across that whole year range, merges them onto a shared
time/lat/lon grid, and writes the result as a single zarr store named after
``output_conventions.filename``, next to ``raw`` in the same version tree.

Everything is driven by the config: ``version``, ``temporal_resolution`` (only
the one set is built), ``variables`` (a list, or ``all`` to take every variable
present on disk), and ``chunks``, where -1 means the whole dimension.

The write is incremental either way, so an interrupted run resumes from the last
commit rather than starting over, but what a commit covers follows the chunking
and is picked by ``write_strategy``:

``append`` suits a store chunked shallowly along time. Timesteps are pushed out
in batches, each appended to the last; the batch size defaults to
``zarr_utils.DEFAULT_TIMESTEPS_PER_COMMIT`` and can be set with an optional
``timesteps_per_commit`` key. A batch is streamed out chunk by chunk rather than
held whole, so what bounds peak memory is the number of chunks in flight and the
number of netCDF files left open behind them: the optional ``num_workers`` and
``file_cache_maxsize`` keys, applied by ``configure_runtime``.

``region`` suits a store chunked along the whole time axis, where there is no
append boundary to stop at. The store's metadata and coordinates are written
first as a skeleton, then filled in place one ``block_shape`` sized lat/lon
block at a time, each block spanning the full record and each committed on its
own. A block is held in memory whole, so peak memory here is the block itself.
"""

import logging
import argparse
import os

from utils.path_utils import (
    format_attrs,
    load_config,
    raw_dir,
    resolve_variables,
    store_path,
    variable_files,
)
from utils.log_utils import (
    setup_logging,
)
from utils.zarr_utils import (
    DEFAULT_WRITE_STRATEGY,
    block_read_chunks,
    build_encoding,
    check_resume,
    check_resume_region,
    commit_batch_size,
    committed_blocks,
    committed_timesteps,
    configure_runtime,
    create_skeleton,
    derive_attrs,
    iter_blocks,
    merge_variables,
    open_repository,
    open_variable,
    resolve_block_shape,
    resolve_chunks,
    resolve_write_strategy,
    store_variables,
    write_by_region,
    write_dataset,
)

LOG = logging.getLogger(__name__)


def open_variables(settings, variables, chunks):
    """Open every variable as a lazy dataset, keyed by variable name.

    Args:
        settings (dict): The loaded configuration.
        variables (list): Variable names to open.
        chunks (dict): Chunk sizes to open the files with.

    Returns:
        dict: Variable name -> single variable dataset.
    """
    datasets = {}
    for variable in variables:
        files = variable_files(settings, variable)
        LOG.info(f'opening {len(files)} files for {variable}')
        datasets[variable] = open_variable(files, chunks)
    return datasets


def build_dataset(settings, variables):
    """Assemble the merged dataset to write.

    Args:
        settings (dict): The loaded configuration.
        variables (list): Variable names to merge.

    Returns:
        tuple: The merged xarray.Dataset, its resolved chunk sizes, and the
            write strategy to use.
    """
    # the chunks a file is opened with are the unit dask reads in, which is the
    # store's own chunking on the append path but the block on the region path.
    # Read unvalidated, since validating the strategy needs the dimension
    # lengths and those are only known once the files are open
    strategy = settings.get('write_strategy', DEFAULT_WRITE_STRATEGY)
    read_chunks = (
        block_read_chunks(settings) if strategy == 'region' else settings['chunks']
    )
    dataset = merge_variables(open_variables(settings, variables, read_chunks))

    # -1 in the config means the whole dimension, which is only known now that
    # the files are open
    chunks = resolve_chunks(settings['chunks'], dataset.sizes)
    strategy = resolve_write_strategy(settings, chunks, dataset.sizes)

    if strategy == 'append':
        # open_mfdataset chunks each file separately, so the time chunks follow
        # the yearly file boundaries until they are squared up here. The region
        # path writes from memory rather than from dask, so its blocks are left
        # chunked the way they were read
        dataset = dataset.chunk(chunks)

    # merge_variables carries the source files' own attributes over; the derived
    # and configured ones go on top of those so upstream provenance survives
    # beside them. Only the first write lays them down
    dataset.attrs.update(derive_attrs(dataset) | format_attrs(settings))

    LOG.info(
        f'merged {len(dataset.data_vars)} variables: '
        f'{dict(dataset.sizes)}, {dataset.nbytes / 1024**4:.2f} TiB'
    )
    LOG.info(f'carrying {len(dataset.attrs)} global attributes')
    LOG.info(f'chunking as {chunks}, write strategy {strategy!r}')
    return dataset, chunks, strategy


def write_append(repository, dataset, chunks, settings):
    """Write the store by appending batches of timesteps.

    Args:
        repository (icechunk.Repository): The repository to write into.
        dataset (xarray.Dataset): The lazy dataset to write.
        chunks (dict): Resolved chunk sizes.
        settings (dict): The loaded configuration.

    Returns:
        int: The number of batches written.

    Raises:
        ValueError: If a partially written store cannot be resumed into.
    """
    batch_size = commit_batch_size(chunks, settings)
    encoding = build_encoding(dataset, chunks, batch_size)

    # a store left behind by an interrupted run is continued rather than redone
    n_written = committed_timesteps(repository)
    if n_written:
        LOG.info(f'store already holds {n_written} timesteps, checking before resuming')
        check_resume(repository, dataset, n_written)
        if n_written >= dataset.sizes['time']:
            LOG.info('store is already complete, nothing to write')
            return 0
        # resume on a batch boundary; a short final batch cannot have been
        # written without the run having finished
        if n_written % batch_size:
            raise ValueError(
                f'store holds {n_written} timesteps, which is not a whole '
                f'number of {batch_size} step batches; delete it to rebuild'
            )
        LOG.info(f'resuming from timestep {n_written}')

    LOG.info(f'writing {batch_size} timesteps per commit')
    return write_dataset(repository, dataset, encoding, batch_size, start=n_written)


def write_region(repository, dataset, chunks, settings, variables):
    """Write the store as a skeleton, then fill it block by block.

    Args:
        repository (icechunk.Repository): The repository to write into.
        dataset (xarray.Dataset): The lazy dataset to write.
        chunks (dict): Resolved chunk sizes.
        settings (dict): The loaded configuration.
        variables (list): Variable names to write, in a stable order.

    Returns:
        int: The number of blocks written.

    Raises:
        ValueError: If a partially filled store cannot be resumed into.
    """
    block_shape = resolve_block_shape(settings, chunks, dataset.sizes)
    blocks = iter_blocks(variables, dataset.sizes, block_shape)
    LOG.info(f'blocking as {block_shape}: {len(blocks)} blocks')

    # build_encoding also clears the stale netCDF encoding off every variable,
    # which a region write needs just as much as the skeleton does
    encoding = build_encoding(dataset, chunks, dataset.sizes['time'])

    if store_variables(repository):
        LOG.info('store already exists, checking before resuming')
        check_resume_region(repository, dataset, chunks)
        done = committed_blocks(repository)
        # a block is its own resume key, so changing block_shape between runs
        # is safe but wasteful: nothing already written matches, and all of it
        # is written again
        stray = done - set(blocks)
        if stray:
            LOG.warning(
                f'{len(stray)} committed blocks do not match the configured '
                f'block_shape; that work will be redone'
            )
        LOG.info(f'{len(done)} of {len(blocks)} blocks already committed')
    else:
        create_skeleton(repository, dataset, encoding)
        done = set()

    return write_by_region(repository, dataset, blocks, done)


def main(settings):

    os.makedirs(settings['directories']['logs'], exist_ok=True)
    log_file = os.path.join(settings['directories']['logs'], settings['log_file'])
    setup_logging(log_file)

    # set before anything opens a file or builds a graph, since both settings
    # only take effect for work started after them
    configure_runtime(settings)

    path = store_path(settings)
    LOG.info(
        f'building zarr store for GLEAM {settings["version"]} '
        f'({settings["temporal_resolution"]}, variables: {settings["variables"]})'
    )
    LOG.info(f'reading from {raw_dir(settings)}')
    LOG.info(f'writing to {path}')

    # expand 'variables: all' against what was actually downloaded
    variables = resolve_variables(settings)
    LOG.info(f'merging {len(variables)} variables: {variables}')

    dataset, chunks, strategy = build_dataset(settings, variables)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    repository = open_repository(path)

    if strategy == 'append':
        written = write_append(repository, dataset, chunks, settings)
        LOG.info(f'wrote {written} batches to {path}')
    else:
        written = write_region(repository, dataset, chunks, settings, variables)
        LOG.info(f'wrote {written} blocks to {path}')

    LOG.info('done :-)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='build an icechunk zarr store from raw gleam netcdf files.'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file.',
    )
    args = parser.parse_args()
    settings = load_config(args.config)
    main(settings)
