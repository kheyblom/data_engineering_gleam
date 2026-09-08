"""Configuration loading and path construction for the GLEAM data engineering scripts.

The download step lays raw netCDF files out as
``<download>/<version>/raw/<temporal_resolution>/<variable>/<file>.nc``, with the
version written as ``v_4_3_a`` rather than ``v4.3a``. Everything here builds
paths against that layout, so the same config drives both the download and the
zarr conversion, and derived zarr stores are written to a ``zarr`` directory
sitting alongside ``raw`` in the same version tree.
"""

from __future__ import annotations

import os
import re
import glob

import yaml # type: ignore

# 'v4.3a' -> ('4', '3', 'a'), the pieces of the on disk version directory name
VERSION_RE = re.compile(r'^v(?P<major>\d+)\.(?P<minor>\d+)(?P<letter>[a-z])$')

# literal segment of the download tree holding the raw netCDF files
RAW_DIRNAME = 'raw'
# sibling of RAW_DIRNAME holding the zarr stores derived from those files
ZARR_DIRNAME = 'zarr'


def load_config(file_path: str) -> dict:
    """Load YAML configuration from a file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: The loaded configuration.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def format_version(version):
    """Rewrite a GLEAM version for use in a directory or store name.

    Args:
        version (str): Version as written in the config, e.g. 'v4.3a'.

    Returns:
        str: The version with '.' dropped and the parts underscore separated,
            e.g. 'v_4_3_a'.

    Raises:
        ValueError: If the version is not of the form 'v<major>.<minor><letter>'.
    """
    match = VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"cannot parse version {version!r}, expected e.g. 'v4.3a'")
    return 'v_{major}_{minor}_{letter}'.format(**match.groupdict())


def version_root(settings):
    """Root of the local tree for the configured version.

    Args:
        settings (dict): The loaded configuration.

    Returns:
        str: e.g. '<download>/v_4_3_a'.
    """
    return os.path.join(
        settings['directories']['download'], format_version(settings['version'])
    )


def raw_dir(settings):
    """Directory holding the raw netCDF files for the configured resolution.

    Args:
        settings (dict): The loaded configuration.

    Returns:
        str: e.g. '<download>/v_4_3_a/raw/daily'; its subdirectories are the
            variables.
    """
    return os.path.join(
        version_root(settings), RAW_DIRNAME, settings['temporal_resolution']
    )


def format_filename(settings):
    """Render ``output_conventions.filename`` from the rest of the config.

    Every scalar at the top level of the config is available to the template, as
    is every key under ``output_conventions``; ``version`` is substituted in its
    directory form so the store name matches the input tree.

    Args:
        settings (dict): The loaded configuration.

    Returns:
        str: e.g. 'gleam_v_4_3_a_daily_native_0p1x0p1_spatial.zarr'.

    Raises:
        ValueError: If the template refers to a field the config does not define.
    """
    # flatten the config into the fields the template may reference; only scalars
    # are useful in a filename, so nested sections are skipped
    fields = {
        key: value
        for key, value in settings.items()
        if isinstance(value, (str, int, float))
    }
    fields.update(settings.get('output_conventions', {}))
    # the store name carries the same version spelling as the input directories
    fields['version'] = format_version(settings['version'])

    template = settings['output_conventions']['filename']
    try:
        return template.format_map(fields)
    except KeyError as error:
        raise ValueError(
            f'filename template {template!r} refers to {error} '
            f'which is not set in the config'
        ) from None


def store_path(settings):
    """Full path of the zarr store to build.

    Args:
        settings (dict): The loaded configuration.

    Returns:
        str: e.g. '<download>/v_4_3_a/zarr/gleam_v_4_3_a_daily_..._spatial.zarr'.
    """
    return os.path.join(version_root(settings), ZARR_DIRNAME, format_filename(settings))


def discover_variables(settings):
    """List the variables present on disk for the configured resolution.

    Args:
        settings (dict): The loaded configuration.

    Returns:
        list: Sorted variable names, taken from the subdirectory names under
            ``raw_dir``.

    Raises:
        FileNotFoundError: If the raw directory does not exist.
    """
    directory = raw_dir(settings)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'no raw data directory at {directory}')
    return sorted(
        entry
        for entry in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, entry))
    )


def resolve_variables(settings):
    """Return the variables to merge, expanding the ``all`` shorthand.

    Args:
        settings (dict): The loaded configuration.

    Returns:
        list: The variable names to read, in a stable order.

    Raises:
        ValueError: If a variable named in the config has no directory on disk,
            or if no variables were found at all.
    """
    available = discover_variables(settings)
    requested = settings['variables']

    # 'variables: all' takes every variable directory, otherwise it is a list
    if isinstance(requested, str) and requested == 'all':
        variables = available
    else:
        # keep the order the config asks for, but reject anything not downloaded
        missing = [name for name in requested if name not in available]
        if missing:
            raise ValueError(
                f'variables {missing} are not in {raw_dir(settings)}; '
                f'available: {available}'
            )
        variables = list(requested)

    if not variables:
        raise ValueError(f'no variables found in {raw_dir(settings)}')
    return variables


def variable_files(settings, variable):
    """List one variable's netCDF files in time order.

    Filenames are '<variable>_<year>_GLEAM_<version>.nc', so sorting them
    lexically also sorts them by year.

    Args:
        settings (dict): The loaded configuration.
        variable (str): The variable directory to list.

    Returns:
        list: Sorted absolute paths to the netCDF files.

    Raises:
        FileNotFoundError: If the variable directory holds no netCDF files.
    """
    pattern = os.path.join(raw_dir(settings), variable, '*.nc')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'no netCDF files matching {pattern}')
    return files
