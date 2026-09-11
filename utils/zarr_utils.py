"""Helpers for building icechunk backed zarr stores from netCDF inputs.

Either way a store is written, it is written in pieces, each committed to the
icechunk repository before the next one starts. That is what makes a run
restartable, which matters when a store is large enough that a single write will
not fit in one job's walltime. Which piece is the unit depends on how the store
is chunked, and ``write_strategy`` in the config picks between the two:

``append`` is for a store chunked shallowly along time, where a timestep is
cheap to write and the whole spatial grid is not. The dataset is assembled
lazily and pushed out in batches of timesteps, each appended to the last. A
batch is streamed chunk by chunk rather than held whole, so peak memory follows
the chunks in flight, which ``configure_runtime`` bounds. Batch boundaries are
always a whole number of time chunks: appending onto a partially filled chunk
would mean rewriting it, which xarray refuses to do from dask, so only the final
batch is allowed to be short.

``region`` is for a store chunked along the whole time axis, where appending is
impossible -- every chunk spans every timestep, so there is no boundary to
append at and a batched write collapses into one uninterruptible commit. Instead
the store's metadata and coordinates are laid down first as a skeleton, and the
arrays are then filled in place, one lat/lon block at a time across the full
record. A block is read whole into memory rather than streamed, so peak memory
is the block, and the block is deliberately much larger than an output chunk:
the raw files are chunked coarsely in space, so a small block re-reads and
re-decompresses the same source chunk for every tile it overlaps.
"""

from __future__ import annotations

import logging
import re

import dask
from dask.system import CPU_COUNT
import netCDF4
import numpy as np
import xarray as xr
import icechunk
from icechunk.xarray import to_icechunk

LOG = logging.getLogger(__name__)

# timesteps written per commit when the config does not say otherwise; small
# enough to keep a single batch cheap to redo after a failure, large enough that
# commit overhead stays negligible
DEFAULT_TIMESTEPS_PER_COMMIT = 100

# chunk cache slots. The raw files are chunked [12, 57, 113], so one 12 timestep
# deep row covering a whole lat/lon plane is 32 x 32 = 1024 chunks; HDF5 hashes
# chunks into these slots, so it wants comfortably more than that, and a prime
# spreads them evenly.
DEFAULT_CHUNK_CACHE_SLOTS = 2003

# branch every commit is written to
BRANCH = 'main'

# how a store is filled. 'append' extends the time axis batch by batch; 'region'
# lays down a skeleton and fills it block by block. See the module docstring.
DEFAULT_WRITE_STRATEGY = 'append'
WRITE_STRATEGIES = ('append', 'region')

# the region path's resume state lives in the commit messages rather than in the
# store, so that reading it back needs nothing but the history icechunk already
# keeps, and so finalize_gleam_zarr.py does not have to strip build bookkeeping
# out of the attributes it publishes. The pair has to stay in step.
SKELETON_MESSAGE = 'create skeleton'
BLOCK_MESSAGE = 'write {variable} lat[{lat0}:{lat1}) lon[{lon0}:{lon1})'
BLOCK_MESSAGE_RE = re.compile(
    r'^write (?P<variable>\S+) '
    r'lat\[(?P<lat0>\d+):(?P<lat1>\d+)\) '
    r'lon\[(?P<lon0>\d+):(?P<lon1>\d+)\)$'
)


def configure_runtime(settings):
    """Apply the optional execution settings that bound cost and peak memory.

    None of these change the store that is written, only what it costs to write
    it. Dask's worker count sets how many chunks are held in flight at once;
    xarray's open file cache sets how many netCDF files stay open; and the HDF5
    chunk cache sets how much decompressed data each of those files keeps, which
    is what stops a 12 timestep deep source chunk being decompressed once per
    timestep. All are left to the library default when the config does not set
    them, so an unset ``num_workers`` still follows the cpuset the batch job was
    given.

    The last two multiply: the chunk cache is per open file, so the memory this
    can reach is roughly ``file_cache_maxsize`` x ``chunk_cache_size_mib``.

    Args:
        settings (dict): The loaded configuration.
    """
    num_workers = settings.get('num_workers')
    if num_workers:
        dask.config.set(num_workers=num_workers)

    file_cache_maxsize = settings.get('file_cache_maxsize')
    if file_cache_maxsize:
        xr.set_options(file_cache_maxsize=file_cache_maxsize)

    # The raw files are chunked 12 timesteps deep but the store is written one
    # timestep at a time, so without a cache big enough to hold a whole 12 deep
    # row of chunks (302 MiB) HDF5 decompresses each one twelve times over.
    # Measured at 7.6x end to end, which dwarfs every other setting here.
    chunk_cache_size_mib = settings.get('chunk_cache_size_mib')
    if chunk_cache_size_mib:
        netCDF4.set_chunk_cache(
            size=chunk_cache_size_mib * 1024**2,
            nelems=settings.get('chunk_cache_slots', DEFAULT_CHUNK_CACHE_SLOTS),
            # keep fully read chunks in preference to partially read ones
            preemption=0.75,
        )

    # log what was resolved rather than what was asked for, so a run killed for
    # running out of memory can be read back against the settings it really used
    LOG.info(
        f'dask workers: {num_workers or CPU_COUNT}'
        f'{"" if num_workers else " (detected)"}, '
        f'open file cache: {xr.get_options()["file_cache_maxsize"]} files, '
        f'hdf5 chunk cache: {chunk_cache_size_mib or "default"} MiB per file'
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


def build_encoding(dataset, chunks, time_coord_chunk):
    """Build zarr encoding for every variable, replacing the netCDF encoding.

    The encoding xarray carries over from the netCDF files describes HDF5
    settings (``zlib``, ``chunksizes``, ...) that the zarr backend rejects, so it
    is dropped and rebuilt here.

    Args:
        dataset (xarray.Dataset): The dataset about to be written.
        chunks (dict): Resolved chunk sizes.
        time_coord_chunk (int): Chunk length for the time coordinate. The append
            path passes the commit batch size, so that each append lands on a
            chunk boundary; the region path has no batches and passes the whole
            axis.

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

        # an index coordinate is a label rather than a field: anything that opens
        # the store reads it whole, and it is small enough that splitting it buys
        # nothing. Without this it inherits the chunking of the data it indexes,
        # which at chunks.lat = 20 would give the 1800 long lat coordinate 90
        # chunks. Harmless at chunks.lat = -1, which is why it went unnoticed.
        if variable.ndim == 1 and variable.dims[0] == name:
            encoding[name]['chunks'] = (variable.sizes[name],)

        if name == 'time':
            encoding[name]['chunks'] = (time_coord_chunk,)
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


def derive_attrs(dataset):
    """Global attributes that can only be read off the data itself.

    Kept apart from the config's ``attrs`` so the coverage and grid figures
    always describe what was actually written, rather than a number someone has
    to remember to update. Callers merge the config on top, so any of these can
    still be overridden by hand.

    Args:
        dataset (xarray.Dataset): The dataset being written or inspected.

    Returns:
        dict: ACDD coverage and geospatial attributes.
    """
    time = dataset['time'].values
    lat = dataset['lat'].values
    lon = dataset['lon'].values
    # the grid is regular, so one step describes it; abs() because lat runs
    # north to south and the resolution is not a signed quantity
    lat_step = abs(float(lat[1] - lat[0]))
    lon_step = abs(float(lon[1] - lon[0]))
    # the coordinates are float32, so widening them to python floats exposes the
    # representation error (89.94999999998977 for a cell centred on 89.95).
    # Four decimals is finer than the 0.1 degree grid and coarser than the
    # error, so it reports the grid the files describe rather than the float
    return {
        'time_coverage_start': str(time.min())[:10],
        'time_coverage_end': str(time.max())[:10],
        'time_coverage_resolution': 'P1D',
        'geospatial_lat_min': round(float(lat.min()), 4),
        'geospatial_lat_max': round(float(lat.max()), 4),
        'geospatial_lon_min': round(float(lon.min()), 4),
        'geospatial_lon_max': round(float(lon.max()), 4),
        'geospatial_lat_resolution': round(lat_step, 4),
        'geospatial_lon_resolution': round(lon_step, 4),
    }


def open_repository(path):
    """Open the icechunk repository at a path, creating it if it is not there.

    Args:
        path (str): Directory holding the store.

    Returns:
        icechunk.Repository: The opened repository.
    """
    storage = icechunk.local_filesystem_storage(path)
    return icechunk.Repository.open_or_create(storage)


def open_existing_repository(path):
    """Open an icechunk repository that must already exist.

    ``open_repository`` creates one if it is absent, which is right for the
    build and wrong for anything administrative: a mistyped path would create an
    empty repository beside the real store, and every subsequent report --
    a garbage collection that freed nothing, a store that verified clean --
    would be true of that empty repository rather than of the store.

    Args:
        path (str): Directory holding the store.

    Returns:
        icechunk.Repository: The opened repository.

    Raises:
        Exception: If there is no repository at the path.
    """
    return icechunk.Repository.open(icechunk.local_filesystem_storage(path))


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


def resolve_write_strategy(settings, chunks, sizes):
    """Pick the write strategy and check it suits the configured chunking.

    The two strategies are not interchangeable, and getting the pairing wrong
    fails quietly rather than loudly, which is why it is checked here. An
    ``append`` build of a store whose time chunk spans the whole record
    degenerates to a single batch: ``commit_batch_size`` rounds the target up to
    one whole chunk, ``write_dataset`` runs one iteration, and the run becomes
    one uninterruptible commit that a walltime kill loses entirely. A ``region``
    build of a shallowly chunked store is merely wasteful, reading the whole
    record into memory to write chunks one timestep deep.

    Args:
        settings (dict): The loaded configuration.
        chunks (dict): Resolved chunk sizes.
        sizes (Mapping): Length of each dimension, e.g. ``dataset.sizes``.

    Returns:
        str: The strategy to use, one of ``WRITE_STRATEGIES``.

    Raises:
        ValueError: If the strategy is unknown, or does not match the chunking.
    """
    strategy = settings.get('write_strategy', DEFAULT_WRITE_STRATEGY)
    if strategy not in WRITE_STRATEGIES:
        raise ValueError(
            f'unknown write_strategy {strategy!r}; expected one of '
            f'{list(WRITE_STRATEGIES)}'
        )

    whole_axis = chunks['time'] >= sizes['time']
    if strategy == 'append' and whole_axis:
        raise ValueError(
            f"write_strategy 'append' needs a time chunk shorter than the "
            f'record, but chunks.time resolves to {chunks["time"]} against '
            f'{sizes["time"]} timesteps; the whole build would be one commit '
            f"with no resume. Use write_strategy 'region'"
        )
    if strategy == 'region' and not whole_axis:
        raise ValueError(
            f"write_strategy 'region' fills whole time chunks in place, but "
            f'chunks.time resolves to {chunks["time"]} against '
            f'{sizes["time"]} timesteps; set chunks.time to -1 or use '
            f"write_strategy 'append'"
        )
    return strategy


def block_read_chunks(settings):
    """Chunks to open the raw files with on the region path.

    The store's own chunking is the wrong thing to read with here. Opening
    644 files as 20x20 tiles would build roughly ten million dask chunks before
    a single byte is read, and each tile would decompress the 57x113 source
    chunk it sits inside. The block is the read unit instead, so one source
    chunk is decompressed once for all the tiles it covers.

    Args:
        settings (dict): The loaded configuration.

    Returns:
        dict: Chunk sizes for ``open_mfdataset``, whole along time because a
            block spans the record and ``open_mfdataset`` chunks per file.
    """
    block_shape = settings.get('block_shape') or {}
    return {
        'time': -1,
        'lat': block_shape.get('lat', -1),
        'lon': block_shape.get('lon', -1),
    }


def resolve_block_shape(settings, chunks, sizes):
    """Resolve the lat/lon block the region path reads and commits in.

    Args:
        settings (dict): The loaded configuration; ``block_shape`` may set
            either dimension, and -1 or an absent key means the whole one.
        chunks (dict): Resolved chunk sizes.
        sizes (Mapping): Length of each dimension, e.g. ``dataset.sizes``.

    Returns:
        dict: Block length per spatial dimension.

    Raises:
        ValueError: If a block is not a whole number of output chunks.
    """
    block_shape = settings.get('block_shape') or {}
    resolved = {}
    for dimension in ('lat', 'lon'):
        size = block_shape.get(dimension, -1)
        size = sizes[dimension] if size == -1 else size
        # a region write addresses the store's chunk grid directly, so a block
        # that is not a whole number of chunks would land mid chunk and force
        # zarr to read, patch and rewrite chunks the next block also touches
        if size % chunks[dimension]:
            raise ValueError(
                f'block_shape.{dimension} of {size} is not a whole number of '
                f'{chunks[dimension]} cell chunks'
            )
        resolved[dimension] = min(size, sizes[dimension])
    return resolved


def iter_blocks(variables, sizes, block_shape):
    """Every block of the region build, in the order it should be written.

    Variable major, so one variable's files stay open across all its blocks
    rather than being reopened once per block.

    Args:
        variables (list): Variable names to write.
        sizes (Mapping): Length of each dimension, e.g. ``dataset.sizes``.
        block_shape (dict): Resolved block length per spatial dimension.

    Returns:
        list: ``(variable, lat0, lat1, lon0, lon1)`` tuples, each of which is
            both the unit of work and its own resume key.
    """
    blocks = []
    for variable in variables:
        for lat0 in range(0, sizes['lat'], block_shape['lat']):
            lat1 = min(lat0 + block_shape['lat'], sizes['lat'])
            for lon0 in range(0, sizes['lon'], block_shape['lon']):
                lon1 = min(lon0 + block_shape['lon'], sizes['lon'])
                blocks.append((variable, lat0, lat1, lon0, lon1))
    return blocks


def block_message(block):
    """Render a block as the commit message that records it.

    Args:
        block (tuple): ``(variable, lat0, lat1, lon0, lon1)``.

    Returns:
        str: The commit message ``committed_blocks`` parses back.
    """
    variable, lat0, lat1, lon0, lon1 = block
    return BLOCK_MESSAGE.format(
        variable=variable, lat0=lat0, lat1=lat1, lon0=lon0, lon1=lon1
    )


def create_skeleton(repository, dataset, encoding):
    """Lay down the store's metadata and coordinates, with no data chunks.

    ``compute=False`` writes the group, every array's metadata and every
    variable already in memory, and defers only the dask backed ones -- which
    here is all of the data. The result is a store of the right shape and
    chunking that ``write_by_region`` can then fill in place.

    Args:
        repository (icechunk.Repository): The repository to write into.
        dataset (xarray.Dataset): The lazy dataset whose shape to lay down.
        encoding (dict): Encoding for the arrays being created.

    Returns:
        str: The snapshot id of the commit.
    """
    # a dask backed coordinate would have its array created and left empty,
    # since compute=False defers exactly the writes that are not yet in memory
    dataset = dataset.assign_coords(
        {name: coordinate.load() for name, coordinate in dataset.coords.items()}
    )

    # open_mfdataset leaves one dask chunk per file along time, so a store chunk
    # spanning the whole record would straddle several of them and xarray
    # refuses the write rather than risk two dask tasks writing one chunk. No
    # data is written here, but the check runs anyway, so present the time axis
    # as the single chunk the region strategy requires it to be. This is a
    # graph operation on a lazy dataset and reads nothing.
    dataset = dataset.chunk({'time': -1})

    session = repository.writable_session(branch=BRANCH)
    dataset.to_zarr(
        session.store,
        mode='w',
        encoding=encoding,
        consolidated=False,
        zarr_format=3,
        compute=False,
    )
    snapshot = session.commit(SKELETON_MESSAGE)
    LOG.info(f'created skeleton as {snapshot}')
    return snapshot


def store_variables(repository):
    """Data variables already present in the store.

    Args:
        repository (icechunk.Repository): The repository to inspect.

    Returns:
        set: The store's data variable names, empty if nothing is written yet.
    """
    session = repository.readonly_session(branch=BRANCH)
    try:
        stored = xr.open_zarr(session.store, consolidated=False)
    except Exception:
        # a freshly created repository has an empty root group and no arrays
        return set()
    return set(stored.data_vars)


def committed_blocks(repository):
    """The blocks a previous run already wrote, read back from the history.

    Progress lives in the commit messages rather than in the store so that
    resuming needs nothing but the history icechunk keeps anyway, and so that
    ``finalize_gleam_zarr.py`` does not have to strip build bookkeeping out of
    the attributes it publishes.

    Args:
        repository (icechunk.Repository): The repository to inspect.

    Returns:
        set: ``(variable, lat0, lat1, lon0, lon1)`` tuples already written.
    """
    blocks = set()
    for snapshot in repository.ancestry(branch=BRANCH):
        match = BLOCK_MESSAGE_RE.match(snapshot.message)
        if match is None:
            continue
        blocks.add(
            (
                match['variable'],
                int(match['lat0']),
                int(match['lat1']),
                int(match['lon0']),
                int(match['lon1']),
            )
        )
    return blocks


def check_resume_region(repository, dataset, chunks):
    """Check a partially filled store matches the dataset being written.

    The region path never appends, so a mismatch would not fail at the write:
    it would quietly overwrite part of one store with data belonging to
    another. This is the guard ``check_resume`` is for the append path.

    Args:
        repository (icechunk.Repository): The repository holding the store.
        dataset (xarray.Dataset): The dataset about to be written.
        chunks (dict): Resolved chunk sizes.

    Raises:
        ValueError: If the store's variables, coordinates or chunk grid do not
            match.
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

    # the whole axis, not an overlap: a region build creates every coordinate up
    # front, so anything short of equality is a different dataset
    for name in ('time', 'lat', 'lon'):
        if not stored[name].equals(dataset[name]):
            raise ValueError(
                f'the {name} coordinate in the store does not match the input '
                f'files; delete the store to rebuild it'
            )

    for name in sorted(expected):
        shape = tuple(chunks[dimension] for dimension in stored[name].dims)
        if tuple(stored[name].encoding['chunks']) != shape:
            raise ValueError(
                f'{name} in the store is chunked '
                f'{tuple(stored[name].encoding["chunks"])} but this run would '
                f'write {shape}; delete the store to rebuild it'
            )


def read_into_buffer(array):
    """Read a lazy array into one preallocated buffer, a time chunk at a time.

    ``.load()`` would be the obvious thing here and it is the wrong one at this
    size. Dask assembles a block by holding every input chunk and allocating the
    concatenated output alongside them, so the block is transiently **doubled**
    at the moment it comes together -- 45 GiB becomes 90 GiB with no warning.
    Against a job's memory cgroup that does not fail cleanly: the kernel spends
    itself on page reclaim (measured 46 minutes of system time against 2 of
    user) and the allocation eventually fails inside HDF5, which reports it as
    the uninformative ``NetCDF: HDF error``.

    Filling a buffer instead makes peak memory the block plus one time chunk,
    and the chunk boundaries are the file boundaries, so each read is still one
    contiguous hyperslab out of one file.

    Args:
        array (xarray.DataArray): The lazy, dask backed block to read.

    Returns:
        numpy.ndarray: The block's values, CF decoded.
    """
    values = np.empty(array.shape, dtype=array.dtype)
    begin = 0
    for step in array.chunks[array.dims.index('time')]:
        values[begin:begin + step] = array.isel(
            time=slice(begin, begin + step)
        ).values
        begin += step
    return values


def write_by_region(repository, dataset, blocks, done=frozenset()):
    """Fill a skeleton store block by block, committing each one.

    Each block is one variable's full time record over a lat/lon tile. It is
    read into memory whole -- the point of the strategy is that the source
    chunks underneath it are decompressed once rather than once per output
    chunk -- so peak memory here is the block, not the chunks in flight. See
    ``read_into_buffer`` for why it is not read with ``.load()``.

    Args:
        repository (icechunk.Repository): The repository to write into.
        dataset (xarray.Dataset): The lazy dataset to read blocks from.
        blocks (list): Every block of the build, from ``iter_blocks``.
        done (Container): Blocks a previous run already committed.

    Returns:
        int: The number of blocks written by this call.
    """
    n_time = dataset.sizes['time']
    remaining = [block for block in blocks if block not in done]
    if not remaining:
        LOG.info('store is already complete, nothing to write')
        return 0
    LOG.info(f'{len(remaining)} blocks to write of {len(blocks)}')

    n_written = 0
    for position, block in enumerate(remaining, start=1):
        variable, lat0, lat1, lon0, lon1 = block
        message = block_message(block)
        region = {
            'time': slice(0, n_time),
            'lat': slice(lat0, lat1),
            'lon': slice(lon0, lon1),
        }

        array = dataset[variable].isel(lat=region['lat'], lon=region['lon'])
        LOG.info(
            f'block {position}/{len(remaining)}: reading {message} '
            f'({array.nbytes / 1024**3:.1f} GiB)'
        )
        values = read_into_buffer(array)

        # a region write addresses arrays that already exist, so the index
        # coordinates naming the region are not part of what is written
        data = xr.Dataset({variable: (array.dims, values, dict(array.attrs))})

        session = repository.writable_session(branch=BRANCH)
        to_icechunk(data, session, region=region)
        snapshot = session.commit(message)
        LOG.info(f'committed {message} as {snapshot}')

        # drop the block before the next iteration reads one. The read happens
        # while these are still bound, so without this the outgoing and the
        # incoming block coexist and peak memory is two blocks rather than one
        # -- the same doubling read_into_buffer exists to avoid, moved one
        # level out.
        del values, data
        n_written += 1
    return n_written
