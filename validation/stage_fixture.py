"""Stage a miniature GLEAM raw tree for the pipeline test.

The pipeline's fixture used to be symlinks to whole year files at the full
1800x3600 grid. That is faithful and unusable: a build is minutes per variable
and the whole test matrix wants a batch job, which is how a test stops being
run. This writes a subset instead -- real files, real attributes, real fill
sentinel, a hundredth of the cells -- so the test runs in about a minute on a
login node.

What is subset and what is preserved is deliberate:

- **The window holds land and ocean.** Roughly 40N..30N by 0..20E, which is
  land on 48% of its cells. An all-land window would never leave an ocean tile
  absent, and the absent-tile checks are a large part of what there is to test.
- **The shape divides the chunk grid.** 100 x 200 is a whole number of 20 x 20
  chunks. The verifier skips its absent-chunk checks outright when it is not,
  and a fixture that quietly skips the checks it exists to exercise is worse
  than no fixture.
- **Two years**, so there are file seams to land on and several commit batches.
- **Variable attributes, units and the -999 fill come through verbatim**, so
  everything downstream reads the metadata it would read in production. The
  variables are chosen so that Ep == Ep_aero + Ep_rad can be checked across
  sibling stores, and E is present as a total whose components are not, which
  is the case that has to skip gracefully rather than fail.
- **Time chunking stays 12 deep**, as upstream, so the append path meets the
  same read pattern it does in production, in miniature.

Idempotent: a file that is already staged is left alone unless --restage.
"""

from __future__ import annotations

import argparse
import os
import sys

import netCDF4
import numpy as np

from utils.path_utils import (
    format_version,
    load_config,
    raw_dir,
    variable_files,
)

# the window, as indices into the full 1800 x 3600 grid. See the module
# docstring for why this one
LAT = slice(500, 600)
LON = slice(1800, 2000)

# Ep and its two components make the identity checkable across sibling stores;
# E is a total whose components are absent, which must skip rather than fail
VARIABLES = ('E', 'Ep', 'Ep_aero', 'Ep_rad')

YEARS = ('1980', '1981')

# upstream chunks 12 deep in time; the spatial extents are scaled to the window
CHUNKING = (12, 50, 100)


def parse_args():
    """Parse the command line.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source-config',
        default='config/config_zarr_spatial.yaml',
        help='Config naming the real raw tree to subset.',
    )
    parser.add_argument(
        '--fixture-config',
        default='config/config_zarr_fixture_spatial.yaml',
        help='Config naming the fixture tree to write.',
    )
    parser.add_argument(
        '--restage', action='store_true', help='Rewrite files that already exist.'
    )
    return parser.parse_args()


def subset_file(source, destination, variable):
    """Write one year file's window as a standalone netCDF.

    Args:
        source (str): The real file to read.
        destination (str): The fixture file to write.
        variable (str): The variable it holds.

    Returns:
        int: Timesteps written.
    """
    read = netCDF4.Dataset(source)
    # the sentinel is the point: masking would hand back a masked array and the
    # fill would stop being the -999 everything downstream compares against
    read.set_auto_mask(False)

    values = read.variables[variable][:, LAT, LON]
    lat = read.variables['lat'][LAT]
    lon = read.variables['lon'][LON]
    time = read.variables['time'][:]

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    write = netCDF4.Dataset(destination, 'w', format='NETCDF4')
    write.createDimension('time', None)
    write.createDimension('lat', len(lat))
    write.createDimension('lon', len(lon))

    for name, source_values in (('time', time), ('lat', lat), ('lon', lon)):
        coordinate = write.createVariable(name, read.variables[name].dtype, (name,))
        coordinate.setncatts(
            {k: read.variables[name].getncattr(k) for k in read.variables[name].ncattrs()}
        )
        coordinate[:] = source_values

    attrs = {
        k: read.variables[variable].getncattr(k)
        for k in read.variables[variable].ncattrs()
    }
    # _FillValue is not settable after creation, so it is passed to the
    # constructor and kept out of the bulk attribute copy
    fill = attrs.pop('_FillValue', np.float32(-999.0))
    array = write.createVariable(
        variable,
        'f4',
        ('time', 'lat', 'lon'),
        zlib=True,
        complevel=1,
        chunksizes=CHUNKING,
        fill_value=fill,
    )
    array.setncatts(attrs)
    array[:] = values

    # the global attributes are GLEAM's provenance and every store carries them
    write.setncatts({k: read.getncattr(k) for k in read.ncattrs()})
    write.close()
    read.close()
    return len(time)


def main():
    """Stage the fixture tree.

    Returns:
        int: 0 on success.
    """
    args = parse_args()
    source_settings = load_config(args.source_config)
    fixture_settings = load_config(args.fixture_config)

    destination_root = os.path.join(
        fixture_settings['directories']['download'],
        format_version(fixture_settings['version']),
        'raw',
        fixture_settings['temporal_resolution'],
    )
    print(f'source  {raw_dir(source_settings)}')
    print(f'fixture {destination_root}')
    print(f'window  lat[{LAT.start}:{LAT.stop}] lon[{LON.start}:{LON.stop}], '
          f'{len(VARIABLES)} variables, {len(YEARS)} years')

    staged = skipped = 0
    for variable in VARIABLES:
        sources = {
            os.path.basename(path): path
            for path in variable_files(source_settings, variable)
        }
        for year in YEARS:
            name = next((n for n in sources if f'_{year}_' in n), None)
            if name is None:
                raise SystemExit(f'no {year} file for {variable} in the source tree')
            destination = os.path.join(destination_root, variable, name)
            if os.path.exists(destination) and not args.restage:
                skipped += 1
                continue
            steps = subset_file(sources[name], destination, variable)
            size = os.path.getsize(destination) / 1024**2
            print(f'  {variable:8s} {year}  {steps} steps, {size:.1f} MiB')
            staged += 1

    print(f'{staged} files staged, {skipped} already present')
    return 0


if __name__ == '__main__':
    sys.exit(main())
