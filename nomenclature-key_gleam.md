# Nomenclature Key — data_engineering_gleam

Maps GLEAM's published nomenclature onto the authoritative
[nomenclature_data.md](/glade/u/home/kheyblom/work/style_guides/nomenclature_data.md),
as required by the data engineering style guide.

**This file is machine read.** `utils/nomenclature.py` parses the two tables
below and is the only place the pipeline learns what a variable is called, so
the key and the stores cannot drift apart. Edit the tables, not the code.
Columns are matched by their header name, so they may be reordered but not
renamed.

Every `unit_conversion` here is `none`: GLEAM already publishes these quantities
in the canonical units, spelled differently. No data value is altered anywhere
in this mapping.

## Variable Key

| original_variable_name | original_units | canonical_name | canonical_units | canonical_long_name | unit_conversion | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `E` | `mm.day-1` | `evaporation` | `mm d-1` | total evaporation flux | none | spelling only |
| `Eb` | `mm.day-1` | `evaporation_bare_soil` | `mm d-1` | bare soil evaporation flux | none | spelling only |
| `Ec` | `mm.day-1` | `condensation` | `mm d-1` | condensation flux | none | spelling only |
| `Ei` | `mm.day-1` | `evaporation_interception_loss` | `mm d-1` | evaporation interception loss | none | spelling only |
| `Ep` | `mm.day-1` | `potential_evaporation` | `mm d-1` | potential evaporation flux | none | spelling only |
| `Ep_aero` | `mm.day-1` | `potential_evaporation_aerodynamic` | `mm d-1` | potential evaporation flux from the aerodynamic component | none | spelling only |
| `Ep_rad` | `mm.day-1` | `potential_evaporation_radiative` | `mm d-1` | potential evaporation flux from the radiative component | none | spelling only |
| `Es` | `mm.day-1` | `snow_sublimation` | `mm d-1` | snow sublimation flux | none | spelling only |
| `Et` | `mm.day-1` | `transpiration` | `mm d-1` | transpiration flux | none | spelling only |
| `Ew` | `mm.day-1` | `evaporation_open_water` | `mm d-1` | open-water evaporation flux | none | spelling only |
| `H` | `W.m-2` | `sensible_heat_flux` | `W m-2` | sensible heat flux | none | spelling only |
| `S` | `-` | `evaporative_stress` | `unitless` | evaporative stress ratio | none | dimensionless either way; `-` is not a udunits token |
| `SMrz` | `m3.m-3` | `soil_moisture_root_zone` | `m3 m-3` | root-zone soil moisture content | none | spelling only |
| `SMs` | `m3.m-3` | `soil_moisture_surface` | `m3 m-3` | surface soil moisture content | none | spelling only |

`original_units` is what GLEAM v4.3a publishes in the netCDF files; it is
recorded here for the reader. The migration and the build both read the
*actual* upstream strings off the data rather than from this table, so a
store's `original_*` attributes cannot be wrong even if this column is.

## Temporal Frequency Key

| original_frequency_name | canonical_name | canonical_long_name |
| --- | --- | --- |
| `daily` | `day` | daily average |

`canonical_long_name` is what `nomenclature_data.md` calls the frequency, and it
is what appears in a store's `title` and `summary` -- 'GLEAM v4.3a daily average
total evaporation flux' rather than the bare token 'day', which reads as a
truncation in prose. The token itself is for the filename and the
`temporal_frequency` attribute.

`temporal_resolution: daily` in every config under `config/` names the raw
directory on disk (`<download>/v_4_3_a/raw/daily/`), which is GLEAM's own
spelling and is not renamed. The canonical `day` is what appears in the store
name and in the stores' `temporal_frequency` attribute:

```
<zarr>/spatial/gleam.v_4_3_a.day.native_0p1x0p1.<canonical_name>.zarr
<zarr>/temporal/gleam.v_4_3_a.day.native_0p1x0p1.<canonical_name>.zarr
```

This is the same split `format_version` already makes between `v4.3a` in the
config and `v_4_3_a` in the name.

## Coordinates

`time`, `lat` and `lon` are **not** renamed. `nomenclature_data.md` has no rows
for coordinate variables, and the style guide's naming rules are written against
its Data Variable Key. They keep the CF names GLEAM publishes.
