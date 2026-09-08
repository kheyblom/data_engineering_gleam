# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Converts raw GLEAM (evaporation/soil moisture) netCDF files into a single
icechunk-backed zarr store on NCAR Derecho/GLADE. One script, one config, no
package install step — `gleam_zarr.py` is run directly from the repo root.

## Commands

```bash
uv sync                                             # create/refresh .venv from uv.lock
uv run python gleam_zarr.py --config config/config_zarr.yaml
```

There is no test suite, linter, or CI configured; the
notebook [draft_zarr.ipynb](draft_zarr.ipynb) is exploratory scratch work, not
part of the pipeline. Progress is logged to both stdout and
`logs/gleam_zarr.log` (gitignored, as are all data outputs — `*.nc`, `*.zarr/`,
`figures/`).

Large runs are long enough to need a batch job; the resume behaviour below
exists so a run killed by walltime can simply be relaunched with the same
config.

## Architecture

Everything is derived from the YAML config — no paths, variable names, or
chunk sizes are hardcoded in the pipeline.

### On-disk layout (produced by a separate download step, not in this repo)

```
<directories.download>/<version>/raw/<temporal_resolution>/<variable>/*.nc
<directories.download>/<version>/zarr/<output_conventions.filename>
```

The version is spelled `v4.3a` in the config but `v_4_3_a` on disk;
`format_version` is the single place that translation happens, and the store
name carries the same directory spelling. `variables: all` expands against the
directories actually present under `raw/<temporal_resolution>/`, and a
config-listed variable with no directory is an error rather than a skip.

### Incremental write model

The dataset is assembled lazily, then pushed out in batches of timesteps, each
committed to the icechunk repo before the next starts, so an interrupted run
resumes from the last commit rather than starting over.

Batching bounds the unit of work, not the resident memory: each batch is
streamed to the store chunk by chunk, so the `batch.nbytes` figure in the log
(~34 GiB at the current settings) is not a memory figure. Peak memory is set by
the chunks in flight — roughly `num_workers` x one chunk, 124 MiB at
`time: 5` on the 1800x3600 grid — plus the HDF5 chunk cache each open netCDF
file carries (64 MiB by default, `file_cache_maxsize` of them).

Constraints that hold this together — breaking any of them breaks resume:

- Batch size is always a whole number of time chunks (`commit_batch_size`
  rounds `timesteps_per_commit`, default 100, to the nearest multiple of the
  time chunk). xarray cannot append onto a partially filled chunk from dask, so
  only the *final* batch may be short.
- The `time` coordinate is encoded with one chunk per batch, so each append
  lands on a chunk boundary.
- On resume, `check_resume` compares stored variables and the overlapping
  timesteps against the incoming dataset, and a store whose length is not a
  multiple of the batch size is rejected — the fix is to delete and rebuild,
  never to patch around it.
- Time encoding (`days since 1900-01-01`, proleptic_gregorian, int64) is pinned
  so every appended batch encodes identically.

### Execution settings

`num_workers` and `file_cache_maxsize` are the two config keys that change what
a run costs rather than what it produces; both are optional and fall back to the
library default when unset. They are applied by `configure_runtime` before
anything opens a file, since neither takes effect retroactively, and the
resolved values are logged so a job killed for running out of memory can be read
back against what it actually used.

`num_workers` must not exceed the `ncpus` the batch job was given — the config
is authoritative, so the job script is what gets adjusted to match. Left unset,
dask sizes its pool from the cpuset instead. `file_cache_maxsize` only has to
cover the files a single batch touches (two year files per variable at most), so
it is set well below xarray's default of 128.

### Non-obvious details

- `parallel=True` is deliberately omitted from `open_mfdataset`: opening netCDF
  from several threads crashes the HDF5 library in this build. Do not add it.
- The netCDF encoding xarray carries over (`zlib`, `chunksizes`, ...) is
  rejected by the zarr backend, so `build_encoding` clears every variable's
  `.encoding` and rebuilds it rather than overriding keys.
- `chunks: -1` in the config means the whole dimension, resolvable only once
  the files are open (`resolve_chunks`). `open_mfdataset` chunks per file, so
  time chunks follow yearly file boundaries until the explicit `.chunk()`.
- `merge_variables` requires an exact time-axis match across variables and
  merges with `join='exact'`; a mismatch means an incomplete download and would
  otherwise surface as a silently NaN-filled variable.

## Conventions

Single quotes, f-strings for log messages, Google-style docstrings with Args /
Returns / Raises on every non-trivial function. Comments explain *why* a choice
was made (the HDF5 threading note, the chunk-boundary rule), not what the line
does. Module docstrings carry the context needed to read the file.
