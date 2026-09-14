"""Build an icechunk backed zarr store from the raw GLEAM netCDF files.

The raw tree written by the download step holds one directory per variable, each
with one netCDF file per year:
``<download>/<version>/raw/<temporal_resolution>/<variable>/<file>.nc``, with the
version written as ``v_4_3_a`` rather than ``v4.3a``. This script opens one
variable across that whole year range and writes it as its own zarr store,
named after ``output_conventions.filename``, next to ``raw`` in the same version
tree.

**One store holds one variable.** ``--variable`` picks which, and the filename
template renders it, so one config addresses a whole layout -- the 14 stores
that share a chunking -- and the stores of a layout are independent
repositories with no writer to contend over. Run without ``--variable`` the
script builds the layout's whole family one at a time, which is what a small or
test config usually wants; production fans out one process per variable instead.

Everything else is driven by the config: ``version``, ``temporal_resolution``
(only the one set is built), ``variables`` (a list, or ``all`` to take every
variable present on disk), and ``chunks``, where -1 means the whole dimension.

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
    check_time_axis,
    commit_batch_size,
    committed_blocks,
    committed_timesteps,
    configure_runtime,
    create_skeleton,
    derive_attrs,
    describe_variable,
    iter_blocks,
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


def build_dataset(settings, variable, reference):
    """Assemble the one variable dataset to write.

    Args:
        settings (dict): The loaded configuration.
        variable (str): The variable to build a store for.
        reference (str): The family's reference variable, whose time axis this
            one is checked against.

    Returns:
        tuple: The xarray.Dataset, its resolved chunk sizes, and the write
            strategy to use.

    Raises:
        ValueError: If this variable does not cover the same timesteps as the
            reference.
    """
    # the chunks a file is opened with are the unit dask reads in, which is the
    # store's own chunking on the append path but the block on the region path.
    # Read unvalidated, since validating the strategy needs the dimension
    # lengths and those are only known once the files are open
    strategy = settings.get('write_strategy', DEFAULT_WRITE_STRATEGY)
    read_chunks = (
        block_read_chunks(settings) if strategy == 'region' else settings['chunks']
    )

    files = variable_files(settings, variable)
    # nothing merges the variables any more, so nothing would notice one of them
    # being a year short. Each build pins itself to the same reference instead
    check_time_axis(variable, files, reference, variable_files(settings, reference))

    LOG.info(f'opening {len(files)} files for {variable}')
    dataset = open_variable(files, read_chunks)

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

    # open_mfdataset carries the source files' own attributes over; the derived
    # and configured ones go on top of those so upstream provenance survives
    # beside them. Only the first write lays them down
    dataset.attrs.update(
        derive_attrs(dataset)
        | describe_variable(dataset, settings)
        | format_attrs(settings)
    )

    LOG.info(
        f'{variable}: {dict(dataset.sizes)}, '
        f'{dataset.nbytes / 1024**4:.2f} TiB logical'
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


def write_region(repository, dataset, chunks, settings, variable):
    """Write the store as a skeleton, then fill it block by block.

    Args:
        repository (icechunk.Repository): The repository to write into.
        dataset (xarray.Dataset): The lazy dataset to write.
        chunks (dict): Resolved chunk sizes.
        settings (dict): The loaded configuration.
        variable (str): The variable the store holds.

    Returns:
        int: The number of blocks written.

    Raises:
        ValueError: If a partially filled store cannot be resumed into.
    """
    block_shape = resolve_block_shape(settings, chunks, dataset.sizes)
    blocks = iter_blocks([variable], dataset.sizes, block_shape)
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


def check_not_published(repository, path, force):
    """Refuse to rebuild a store that has already been published.

    A tag is how a finished store is published here, so a store carrying one is
    something a consumer may already be pinning. Rebuilding it is not a no-op on
    the region path: resume matches blocks by their exact bounds, read back out
    of the commit messages, so a store written with one ``block_shape`` and
    rebuilt with another matches nothing and writes every block again.

    The data would survive that -- the same values are written from the same raw
    files, and the tag is immutable, so anything reading through the tag is
    untouched. What would not survive is the store describing itself honestly:
    the branch tip would carry a ``verification`` attribute earned by a snapshot
    that is no longer the tip. A store must not carry a claim it has not earned,
    so this stops rather than warns.

    Args:
        repository (icechunk.Repository): The repository about to be written.
        path (str): Its path, for the message.
        force (bool): Rebuild anyway.

    Raises:
        ValueError: If the store carries a tag and ``force`` is not set.
    """
    tags = list(repository.list_tags())
    if not tags:
        return
    if force:
        LOG.warning(
            f'{os.path.basename(path)} carries {tags} and is being rebuilt '
            f'anyway; re-verify and re-tag it afterwards, and note that its '
            f'verification attribute describes the tagged snapshot, not this one'
        )
        return
    raise ValueError(
        f'{path} carries {tags}: it has been verified and published, and a '
        f'rebuild would move the branch away from the snapshot that tag names '
        f'while leaving its verification attribute behind. Pass --force to '
        f'rebuild it anyway, or delete the store first'
    )


def build_store(settings, variable, reference, force=False):
    """Build the store holding one variable.

    Args:
        settings (dict): The loaded configuration, with ``variable`` set.
        variable (str): The variable to build.
        reference (str): The family's reference variable for the time axis check.
        force (bool): Rebuild even a published store.

    Returns:
        str: The path written.

    Raises:
        ValueError: If the store is published and ``force`` is not set.
    """
    path = store_path(settings)
    LOG.info(f'writing to {path}')

    dataset, chunks, strategy = build_dataset(settings, variable, reference)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    repository = open_repository(path)
    # a fresh store has no tags, so this only ever stops a rebuild
    check_not_published(repository, path, force)

    if strategy == 'append':
        written = write_append(repository, dataset, chunks, settings)
        LOG.info(f'wrote {written} batches to {path}')
    else:
        written = write_region(repository, dataset, chunks, settings, variable)
        LOG.info(f'wrote {written} blocks to {path}')
    return path


def main(settings, variable=None, force=False):
    """Build one store, or the layout's whole family.

    Args:
        settings (dict): The loaded configuration.
        variable (str): The single variable to build, or None for all of them.
        force (bool): Rebuild stores that have already been published.

    Raises:
        ValueError: If ``variable`` is not one of the family's, or a store is
            published and ``force`` is not set.
    """
    # expand 'variables: all' against what was actually downloaded. The family
    # is needed whichever is built: its first member is the reference every
    # variable's time axis is checked against
    family = resolve_variables(settings)
    if variable is not None and variable not in family:
        raise ValueError(f'{variable!r} is not in the family {family}')
    reference = family[0]
    building = [variable] if variable else family

    os.makedirs(settings['directories']['logs'], exist_ok=True)
    # one log per store when one store is named, so the processes of a fanned
    # out build do not interleave into a single file
    stem, extension = os.path.splitext(settings['log_file'])
    log_file = os.path.join(
        settings['directories']['logs'],
        f'{stem}_{variable}{extension}' if variable else settings['log_file'],
    )
    setup_logging(log_file)

    # set before anything opens a file or builds a graph, since both settings
    # only take effect for work started after them
    configure_runtime(settings)

    LOG.info(
        f'building GLEAM {settings["version"]} ({settings["temporal_resolution"]}), '
        f'{len(building)} of {len(family)} variables: {building}'
    )
    LOG.info(f'reading from {raw_dir(settings)}')

    for name in building:
        # store_path renders the variable, so it is set before the path is built
        settings['variable'] = name
        build_store(settings, name, reference, force)

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
    parser.add_argument(
        '--variable',
        type=str,
        default=None,
        help='Build the store for this one variable. Omitted, every variable of '
        'the layout is built in turn, which is what a test config usually wants; '
        'a production run fans out one process per variable instead.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Rebuild even a store that carries a tag. A tagged store has been '
        'verified and published, and a rebuild moves the branch away from the '
        'snapshot the tag names.',
    )
    args = parser.parse_args()
    settings = load_config(args.config)
    main(settings, args.variable, args.force)
