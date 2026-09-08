"""Helpers for building icechunk backed zarr stores from netCDF inputs.

The stores are written incrementally: the dataset is assembled lazily with dask,
then pushed out in batches of timesteps, each batch committed to the icechunk
repository before the next one starts. That makes a run restartable, which
matters when a store is large enough that a single write will not fit in one
job's walltime. A batch is streamed chunk by chunk rather than held whole, so
peak memory follows the chunks in flight, which ``configure_runtime`` bounds.

Batch boundaries are always a whole number of time chunks. Appending onto a
partially filled chunk would mean rewriting it, which xarray refuses to do from
dask, so only the final batch is allowed to be short.
"""

from __future__ import annotations

import logging

import dask
from dask.system import CPU_COUNT
import numpy as np
import xarray as xr
import icechunk
from icechunk.xarray import to_icechunk

LOG = logging.getLogger(__name__)

# timesteps written per commit when the config does not say otherwise; small
# enough to keep a single batch cheap to redo after a failure, large enough that
# commit overhead stays negligible
DEFAULT_TIMESTEPS_PER_COMMIT = 100

# branch every commit is written to
BRANCH = 'main'


def configure_runtime(settings):
    """Apply the optional execution settings that bound peak memory.

    Neither setting changes the store that is written, only the memory needed to
    write it. Dask's worker count sets how many chunks are held in flight at
    once, and xarray's open file cache sets how many netCDF files stay open,
    each holding an HDF5 chunk cache of its own. Both are left to the library
    default when the config does not set them, so an unset ``num_workers``
    still follows the cpuset the batch job was given.

    Args:
        settings (dict): The loaded configuration.
    """
    num_workers = settings.get('num_workers')
    if num_workers:
        dask.config.set(num_workers=num_workers)

    file_cache_maxsize = settings.get('file_cache_maxsize')
    if file_cache_maxsize:
        xr.set_options(file_cache_maxsize=file_cache_maxsize)

    # log what was resolved rather than what was asked for, so a run killed for
    # running out of memory can be read back against the settings it really used
    LOG.info(
        f'dask workers: {num_workers or CPU_COUNT}'
        f'{"" if num_workers else " (detected)"}, '
        f'open file cache: {xr.get_options()["file_cache_maxsize"]} files'
    )


def resolve_chunks(chunks, sizes):
    """Turn the configured chunk sizes into concrete lengths.

    A chunk of -1 means 'the whole dimension', following the dask convention
    used in the config.

    Args:
        chunks (dict): Chunk size per dimension, as written in the config.
        sizes (Mapping): Length of each dimension, e.g. ``dataset.sizes``.

    Returns:
        dict: Chunk size per dimension, with -1 replaced by the dimension length.

    Raises:
        ValueError: If a chunked dimension is not present in the dataset.
    """
    resolved = {}
    for dimension, size in chunks.items():
        if dimension not in sizes:
            raise ValueError(
                f'config chunks a dimension {dimension!r} that the data does '
                f'not have; dimensions are {list(sizes)}'
            )
        resolved[dimension] = sizes[dimension] if size == -1 else size
    return resolved


def commit_batch_size(chunks, settings):
    """Number of timesteps to write per commit, rounded to whole time chunks.

    Args:
        chunks (dict): Resolved chunk sizes, must include 'time'.
        settings (dict): The loaded configuration; an optional
            ``timesteps_per_commit`` key overrides the default target.

    Returns:
        int: A positive multiple of the time chunk size.
    """
    target = settings.get('timesteps_per_commit', DEFAULT_TIMESTEPS_PER_COMMIT)
    time_chunk = chunks['time']
    # round to the nearest whole number of chunks, but never down to zero
    n_chunks = max(1, round(target / time_chunk))
    return n_chunks * time_chunk


def open_variable(files, chunks):
    """Open one variable's yearly files as a single lazy dataset.

    Args:
        files (list): The variable's netCDF files, in time order.
        chunks (dict): Chunk sizes to open with, as written in the config.

    Returns:
        xarray.Dataset: The files concatenated along time, dask backed.
    """
    return xr.open_mfdataset(
        files,
        # the files are already sorted by year, so concatenate them in the order
        # given rather than paying to inspect coordinates for the ordering
        combine='nested',
        concat_dim='time',
        chunks=chunks,
        # only time varying variables differ between files; taking the lat/lon
        # coordinates from the first file avoids reading them once per file.
        # note parallel=True is deliberately left off: opening netCDF files from
        # several threads crashes the HDF5 library underneath this build
        data_vars='minimal',
        coords='minimal',
        compat='override',
    )


def merge_variables(datasets):
    """Merge the per variable datasets into one, checking they share a time axis.

    Args:
        datasets (dict): Variable name -> single variable dataset.

    Returns:
        xarray.Dataset: All variables on a common set of coordinates.

    Raises:
        ValueError: If the variables do not all cover the same timesteps.
    """
    names = list(datasets)
    reference_name = names[0]
    reference = datasets[reference_name]

    # a mismatch here means an incomplete download, and would otherwise surface
    # much later as a silently reindexed (NaN filled) variable
    for name in names[1:]:
        time = datasets[name]['time']
        if time.sizes['time'] != reference.sizes['time'] or not time.equals(
            reference['time']
        ):
            raise ValueError(
                f'variable {name!r} has {time.sizes["time"]} timesteps that do '
                f'not match {reference_name!r} with '
                f'{reference.sizes["time"]}; the download may be incomplete'
            )

    # join='exact' keeps the merge from quietly padding a mismatched grid
    return xr.merge(
        [datasets[name] for name in names],
        join='exact',
        combine_attrs='drop_conflicts',
    )


def build_encoding(dataset, chunks, batch_size):
    """Build zarr encoding for every variable, replacing the netCDF encoding.

    The encoding xarray carries over from the netCDF files describes HDF5
    settings (``zlib``, ``chunksizes``, ...) that the zarr backend rejects, so it
    is dropped and rebuilt here.

    Args:
        dataset (xarray.Dataset): The dataset about to be written.
        chunks (dict): Resolved chunk sizes.
        batch_size (int): Timesteps per commit; the time coordinate is chunked
            this way so that each append lands on a chunk boundary.

    Returns:
        dict: Encoding to hand to the zarr writer.
    """
    encoding = {}
    for name, variable in dataset.variables.items():
        # chunk each variable along whichever of its dimensions are chunked, and
        # keep any dimension the config does not mention whole
        shape = tuple(
            chunks.get(dimension, variable.sizes[dimension])
            for dimension in variable.dims
        )
        encoding[name] = {'chunks': shape}

        if name == 'time':
            # one chunk per commit, so appending a batch never has to rewrite a
            # partially filled chunk of the time coordinate
            encoding[name]['chunks'] = (batch_size,)
            # pin the calendar so every appended batch is encoded identically
            encoding[name]['units'] = 'days since 1900-01-01'
            encoding[name]['calendar'] = 'proleptic_gregorian'
            encoding[name]['dtype'] = 'int64'
        elif np.issubdtype(variable.dtype, np.floating):
            # store the decoded fill value rather than the netCDF sentinel, so
            # masked cells read back as NaN without a decoding step
            encoding[name]['_FillValue'] = np.nan

    # xarray merges these dicts with each variable's own .encoding, so the stale
    # netCDF settings have to be cleared rather than just overridden
    for variable in dataset.variables.values():
        variable.encoding = {}

    return encoding


def open_repository(path):
    """Open the icechunk repository at a path, creating it if it is not there.

    Args:
        path (str): Directory holding the store.

    Returns:
        icechunk.Repository: The opened repository.
    """
    storage = icechunk.local_filesystem_storage(path)
    return icechunk.Repository.open_or_create(storage)


def committed_timesteps(repository):
    """Number of timesteps already written to the store.

    Args:
        repository (icechunk.Repository): The repository to inspect.

    Returns:
        int: The length of the store's time dimension, or 0 if nothing has been
            written yet.
    """
    session = repository.readonly_session(branch=BRANCH)
    try:
        stored = xr.open_zarr(session.store, consolidated=False)
    except Exception:
        # a freshly created repository has an empty root group and no arrays
        return 0
    return stored.sizes.get('time', 0)


def check_resume(repository, dataset, n_written):
    """Check a partially written store lines up with the dataset being written.

    Guards against resuming into a store built from a different configuration,
    which would otherwise append mismatched data onto the existing timesteps.

    Args:
        repository (icechunk.Repository): The repository holding the store.
        dataset (xarray.Dataset): The dataset about to be written.
        n_written (int): Timesteps already in the store.

    Raises:
        ValueError: If the store's variables or timesteps do not match.
    """
    session = repository.readonly_session(branch=BRANCH)
    stored = xr.open_zarr(session.store, consolidated=False)

    expected = set(dataset.data_vars)
    found = set(stored.data_vars)
    if found != expected:
        raise ValueError(
            f'store already holds variables {sorted(found)} but this run would '
            f'write {sorted(expected)}; delete the store to rebuild it'
        )

    # comparing the whole overlap is cheap next to the data itself, and catches
    # a store built from a different time range or resolution
    if not stored['time'].equals(dataset['time'].isel(time=slice(0, n_written))):
        raise ValueError(
            'the timesteps already in the store do not match the input files; '
            'delete the store to rebuild it'
        )


def write_dataset(repository, dataset, encoding, batch_size, start=0):
    """Write a dataset to the repository in batches, committing each one.

    Args:
        repository (icechunk.Repository): The repository to write into.
        dataset (xarray.Dataset): The lazy dataset to write.
        encoding (dict): Encoding for the initial write; ignored when resuming.
        batch_size (int): Timesteps per batch, a whole number of time chunks.
        start (int): First timestep to write, for resuming a partial store.

    Returns:
        int: The number of batches written.
    """
    n_time = dataset.sizes['time']
    n_batches = 0
    for begin in range(start, n_time, batch_size):
        end = min(begin + batch_size, n_time)
        batch = dataset.isel(time=slice(begin, end))
        session = repository.writable_session(branch=BRANCH)

        LOG.info(
            f'writing timesteps {begin}-{end - 1} of {n_time} '
            f'({end - begin} steps, {batch.nbytes / 1024**3:.1f} GiB)'
        )
        if begin == 0:
            # the first batch lays down the arrays and their chunking
            to_icechunk(batch, session, mode='w', encoding=encoding)
        else:
            # later batches extend the existing arrays along time
            to_icechunk(batch, session, append_dim='time')

        snapshot = session.commit(f'write timesteps {begin}-{end - 1}')
        LOG.info(f'committed timesteps {begin}-{end - 1} as {snapshot}')
        n_batches += 1
    return n_batches
