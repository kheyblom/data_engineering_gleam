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

The write is incremental. Timesteps are pushed out in batches, each committed to
the icechunk repository before the next starts, so an interrupted run resumes
from the last commit rather than starting over. The batch size defaults to
``zarr_utils.DEFAULT_TIMESTEPS_PER_COMMIT`` and can be set with an optional
``timesteps_per_commit`` key in the config.

A batch is streamed out chunk by chunk rather than held whole, so what bounds
peak memory is the number of chunks in flight and the number of netCDF files
left open behind them: the optional ``num_workers`` and ``file_cache_maxsize``
keys, applied by ``configure_runtime``.
"""

import logging
import argparse
import os

from utils.path_utils import (
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
    build_encoding,
    check_resume,
    commit_batch_size,
    committed_timesteps,
    configure_runtime,
    merge_variables,
    open_repository,
    open_variable,
    resolve_chunks,
    write_dataset,
)

LOG = logging.getLogger(__name__)


def open_variables(settings, variables):
    """Open every variable as a lazy dataset, keyed by variable name.

    Args:
        settings (dict): The loaded configuration.
        variables (list): Variable names to open.

    Returns:
        dict: Variable name -> single variable dataset.
    """
    datasets = {}
    for variable in variables:
        files = variable_files(settings, variable)
        LOG.info(f'opening {len(files)} files for {variable}')
        datasets[variable] = open_variable(files, settings['chunks'])
    return datasets


def build_dataset(settings, variables):
    """Assemble the merged, rechunked dataset to write.

    Args:
        settings (dict): The loaded configuration.
        variables (list): Variable names to merge.

    Returns:
        tuple: The merged xarray.Dataset and its resolved chunk sizes.
    """
    dataset = merge_variables(open_variables(settings, variables))

    # -1 in the config means the whole dimension, which is only known now that
    # the files are open
    chunks = resolve_chunks(settings['chunks'], dataset.sizes)
    # open_mfdataset chunks each file separately, so the time chunks follow the
    # yearly file boundaries until they are squared up here
    dataset = dataset.chunk(chunks)

    LOG.info(
        f'merged {len(dataset.data_vars)} variables: '
        f'{dict(dataset.sizes)}, {dataset.nbytes / 1024**4:.2f} TiB'
    )
    LOG.info(f'chunking as {chunks}')
    return dataset, chunks


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

    dataset, chunks = build_dataset(settings, variables)
    batch_size = commit_batch_size(chunks, settings)
    encoding = build_encoding(dataset, chunks, batch_size)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    repository = open_repository(path)

    # a store left behind by an interrupted run is continued rather than redone
    n_written = committed_timesteps(repository)
    if n_written:
        LOG.info(f'store already holds {n_written} timesteps, checking before resuming')
        check_resume(repository, dataset, n_written)
        if n_written >= dataset.sizes['time']:
            LOG.info('store is already complete, nothing to write')
            LOG.info('done :-)')
            return
        # resume on a batch boundary; a short final batch cannot have been
        # written without the run having finished
        if n_written % batch_size:
            raise ValueError(
                f'store holds {n_written} timesteps, which is not a whole '
                f'number of {batch_size} step batches; delete it to rebuild'
            )
        LOG.info(f'resuming from timestep {n_written}')

    LOG.info(f'writing {batch_size} timesteps per commit')
    n_batches = write_dataset(
        repository, dataset, encoding, batch_size, start=n_written
    )
    LOG.info(f'wrote {n_batches} batches to {path}')

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
