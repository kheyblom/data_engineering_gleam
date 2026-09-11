# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Converts raw GLEAM (evaporation/soil moisture) netCDF files into icechunk-backed
zarr stores on NCAR Derecho/GLADE. One script, one config per store, no package
install step — `gleam_zarr.py` is run directly from the repo root.

Two stores are built from the same raw files, differing only in chunking and so
in which read is cheap: `spatial` (`1, 1800, 3600`, one global map per chunk) and
`temporal` (`16802, 20, 20`, one 2x2 degree tile through the whole record).
Neither is derived from the other — both are built from raw and verified against
raw independently — so nothing about one has to be trusted to trust the other.

## Commands

```bash
uv sync                                             # create/refresh .venv from uv.lock
uv run python gleam_zarr.py --config config/config_zarr.yaml
uv run python verify_gleam_zarr.py --config config/config_zarr.yaml

# finalization; writes nothing without --apply, one action per invocation.
# --status first: it reports which steps a store still needs
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --status
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --attrs
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --tag NAME
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --gc
```

There is no test suite, linter, or CI configured; the
notebook [draft_zarr.ipynb](draft_zarr.ipynb) is exploratory scratch work, not
part of the pipeline. `verify_gleam_zarr.py` audits a *finished* store against
the raw files — six phases selectable with `--phases`, read only throughout,
non-zero exit on any failure. It reads the store's chunking and follows it:
the unit of comparison is a *box* that is one chunk of the store being checked,
a global plane in the spatial store and a 20x20 tile through the whole record in
the temporal one. Reading the other layout's box is not a style question — one
global plane out of the temporal store touches every chunk of the variable,
~400 GiB, to return one map. Its `sweep` phase is the only full-coverage check
available: it audits every chunk off the manifest in ~30 s without reading data
(235,228 of them in the spatial store), so it is the thing to run first after
any rebuild.
[TESTING.md](TESTING.md) records how the pipeline was
validated and tuned before the first production build, including the two test
configs that reproduce it, the reasons behind the execution settings, and the
full verification of the v4.3a store. Progress is logged to both stdout and
`logs/gleam_zarr.log` (gitignored, as are all data outputs — `*.nc`, `*.zarr/`,
`figures/`).

`finalize_gleam_zarr.py` is the only script here that **mutates** a finished
store: it writes global attributes, creates a tag, or garbage collects
unreachable objects. Two standing rules. The verifier stays read only — do not
add a destructive flag to it, because it is what proves a finalization step did
no harm, and a tool that both acts and audits answers two questions with one
exit status. And `--gc --apply` is irreversible: run it without `--apply` first,
and re-run `verify_gleam_zarr.py --phases structure,sweep` afterwards.

The write order — `--attrs`, then `--tag`, then `--gc` — is load-bearing, not a
style preference: tags are immutable, so one created before the attributes exist
permanently names a store that does not describe itself. The README carries the
procedure with its gates and its failure path; do not improvise a shorter one.

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

### Two write strategies

`write_strategy` in the config picks how a store is filled, and it must match
the chunking: `resolve_write_strategy` raises on a mismatch rather than letting
it through, because **both mismatches fail quietly**. An `append` build of a
store whose time chunk spans the record silently collapses into one batch and
one commit — `commit_batch_size` rounds the target up to a whole chunk — so a
walltime kill loses the entire run and the `afterany` chain restarts from zero
forever. A `region` build of a shallowly chunked store merely wastes memory.

### Incremental write model: `append`

The dataset is assembled lazily, then pushed out in batches of timesteps, each
committed to the icechunk repo before the next starts, so an interrupted run
resumes from the last commit rather than starting over.

Batching bounds the unit of work, not the resident memory: each batch is
streamed to the store chunk by chunk, so the `batch.nbytes` figure in the log
(~34 GiB at the current settings) is not a memory figure. Peak memory is set by
the chunks in flight — roughly `num_workers` x one chunk, 24.7 MiB at
`time: 1` on the 1800x3600 grid — plus the HDF5 chunk cache each open netCDF
file carries (64 MiB by default, `file_cache_maxsize` of them). At 8 workers
that is under 200 MiB of chunks against a 2 GiB file cache, so the cache, not
the workers, is the larger term.

Constraints that hold this together — breaking any of them breaks resume:

- Batch size is always a whole number of time chunks (`commit_batch_size`
  rounds `timesteps_per_commit`, default 100, to the nearest multiple of the
  time chunk). xarray cannot append onto a partially filled chunk from dask, so
  only the *final* batch may be short. At the configured `time: 1` every
  integer is a whole number of chunks, so the rounding is a no-op and the batch
  is exactly `timesteps_per_commit`; it starts to bite if `time` is raised.
- The `time` coordinate is encoded with one chunk per batch, so each append
  lands on a chunk boundary.
- On resume, `check_resume` compares stored variables and the overlapping
  timesteps against the incoming dataset, and a store whose length is not a
  multiple of the batch size is rejected — the fix is to delete and rebuild,
  never to patch around it.
- Time encoding (`days since 1900-01-01`, proleptic_gregorian, int64) is pinned
  so every appended batch encodes identically.

### Incremental write model: `region`

A store chunked along the whole time axis has no append boundary — every chunk
spans every timestep — so it is built the other way round: `create_skeleton`
writes the group, the array metadata and the coordinates with `compute=False`
(no data chunk at all), and `write_by_region` then fills the arrays in place,
one lat/lon block of the full record at a time, committing each.

- A block is read whole into memory, so **peak memory here is the block**, not
  the chunks in flight. That is the opposite of the append path and the reason
  a region job asks for memory rather than cpus.
- `block_shape` is deliberately far larger than the output chunk. The raw files
  are chunked `[12, 57, 113]`, so a block only 20 cells wide would decompress
  each 57x113 source chunk once per tile it covers — 3.8x at a 20-row block
  against 1.43x at `lat: 200`. Memory is not billed on Derecho and read
  amplification is, so the block buys one down with the other.
- **`block_shape.lon` must be -1.** Volume is not the whole story: a block
  narrower than the globe reads a *strided* subset of the source chunks — 11 of
  every 32 along lon, scattered through the file — instead of contiguous runs of
  32. A 600x1200 block measured 9% cpu and no committed block in eleven minutes
  on `E`, against seconds per block on the sparse `Ec` that the shape was first
  tried on. Same bytes, different locality. Full lon makes every read a
  contiguous run and costs only more resident memory, which is free here.
- `block_shape` must be a whole number of output chunks, or a region write would
  land mid-chunk and force zarr to read, patch and rewrite chunks the next block
  also touches.
- Resume state is the set of blocks already committed, parsed back out of the
  **commit messages** by `committed_blocks` (`BLOCK_MESSAGE` and
  `BLOCK_MESSAGE_RE` have to stay in step). It lives in the history rather than
  in the store so that resuming needs nothing but the ancestry icechunk keeps
  anyway, and so `finalize_gleam_zarr.py --attrs` does not have to strip build
  bookkeeping out of the attributes it publishes.
- Changing `block_shape` between runs is safe but wasteful — no committed block
  matches the new grid, so all of it is written again. The run warns and
  continues; coverage is still complete because every block not in `done` is
  written.
- In the temporal layout a **large fraction of the chunk grid is legitimately
  absent**: a 20x20 tile that is ocean is all-fill for all 16802 timesteps, and
  zarr does not write it. Unlike the spatial store's 25 missing `E` chunks, this
  is the common case rather than the exception, and it is not per-timestep — E's
  all-fill days fall inside chunks that also hold valid days, so they leave no
  hole at all here.

### Execution settings

`num_workers` and `file_cache_maxsize` are the two config keys that change what
a run costs rather than what it produces; both are optional and fall back to the
library default when unset. (`block_shape` changes cost too, but it is not
optional on the region path and it sets the resume granularity as well, so it is
documented with that strategy above.) They are applied by `configure_runtime` before
anything opens a file, since neither takes effect retroactively, and the
resolved values are logged so a job killed for running out of memory can be read
back against what it actually used.

`num_workers` must not exceed the cpus the batch job was actually given. Note
the job script checks this against the affinity mask, not `$NCPUS` or `nproc`:
on a shared develop node PBS reports `NCPUS=1` whatever it granted, and `nproc`
reports `OMP_NUM_THREADS`, which the script pins to 1. Left unset, dask sizes
its pool from the cpuset instead. `file_cache_maxsize` only has to cover the
files a single unit of work touches — two year files per variable on the append
path, every file of one variable on the variable-major region path — so it is
set well below xarray's default of 128 either way.

`chunk_cache_size_mib` matters more than either **on the append path**. The raw
files are chunked `[12, 57, 113]`, twelve timesteps deep, while an append build
writes one timestep at a time, so unless the cache holds a whole twelve-deep row
of chunks (302 MiB) every chunk is decompressed twelve times; setting it to
512 MiB measured 7.6x end to end. On the region path a block is one contiguous
hyperslab per file and every source chunk it touches is decompressed once
regardless, so the cache is not doing that job there. It is per open file either
way, so worst-case memory is `file_cache_maxsize` x `chunk_cache_size_mib`.

### Non-obvious details

- The project is pinned to Python 3.13 (`requires-python = ">=3.13,<3.14"`).
  On 3.14 numpy's temporary-elision optimisation misfires and segfaults on
  `condition |= data == fill_value`, the CF masking in
  `xarray/coding/variables.py`, for any temporary above the elision threshold.
  Every read of a full lat/lon slab goes through that line, so the pipeline
  cannot read a single chunk on 3.14; the same numpy and xarray are fine on
  3.13. Do not raise the Python cap without re-running the tiny test config.
- `parallel=True` is deliberately omitted from `open_mfdataset`: opening netCDF
  from several threads crashes the HDF5 library in this build. Do not add it.
- The netCDF encoding xarray carries over (`zlib`, `chunksizes`, ...) is
  rejected by the zarr backend, so `build_encoding` clears every variable's
  `.encoding` and rebuilds it rather than overriding keys.
- `chunks: -1` in the config means the whole dimension, resolvable only once
  the files are open (`resolve_chunks`). `open_mfdataset` chunks per file, so
  time chunks follow yearly file boundaries until the explicit `.chunk()`.
- Zarr does not write a chunk whose cells are all equal to the fill value, so
  a variable can hold fewer chunks than there are timesteps and still be
  correct — the hole reads back as NaN, which is what an all-fill raw plane
  decodes to. In v4.3a `E` is entirely -999 on 25 days (five 5-day blocks in
  1982 and 1992) while every other variable, including E's own components, has
  data there. That is an upstream GLEAM gap, not a build failure; a short chunk
  count is not a reason to rebuild. `verify_gleam_zarr.py` traces every hole
  back to raw for this reason, and the store's own `known_data_gaps` attribute
  records it so a downstream consumer does not have to rediscover it.
- Every write through `to_icechunk` **forks** the session, and the fork's
  snapshot is not in the branch's ancestry, so a store accumulates one
  unreachable snapshot per commit as a matter of course. A count of unreachable
  snapshots close to the batch count is normal, not damage — read it against
  `len(ancestry(...))` and the object count in `snapshots/` before concluding
  anything. `expire_snapshots` is never needed to clean these up, and reclaims
  essentially no bytes; `garbage_collect` alone is the right tool.
- `merge_variables` requires an exact time-axis match across variables and
  merges with `join='exact'`; a mismatch means an incomplete download and would
  otherwise surface as a silently NaN-filled variable.
- `create_skeleton` rechunks time to `-1` before `to_zarr(compute=False)`. No
  data is written there, but xarray validates chunk alignment anyway and refuses
  a store chunk that straddles several dask chunks — which the per-file time
  chunks `open_mfdataset` leaves behind always do. The rechunk is a graph
  operation on a lazy dataset and reads nothing.
- A region write must not carry the index coordinates that name the region, so
  `write_by_region` drops `time`/`lat`/`lon` from the block before writing. The
  block is `.load()`ed to numpy first, which also sidesteps xarray's
  `safe_chunks` alignment check entirely.
- `build_encoding` keeps 1-D index coordinates as a single chunk. Without that
  they inherit the chunking of the data they index, which at `lat: 20` splits
  the 1800-long `lat` coordinate into 90 chunks. Harmless at `lat: -1`, which is
  why the spatial build never showed it.

## Conventions

Single quotes, f-strings for log messages, Google-style docstrings with Args /
Returns / Raises on every non-trivial function. Comments explain *why* a choice
was made (the HDF5 threading note, the chunk-boundary rule), not what the line
does. Module docstrings carry the context needed to read the file.

### Layout

Both entry points — `gleam_zarr.py` and `verify_gleam_zarr.py` — sit at the
repo root, and `utils/` is importable only because a script's own directory is
what lands on `sys.path`. That is why there is no install step, and it is the
constraint any reorganisation runs into first: moving either script into a
subdirectory breaks `from utils...` immediately.

Considered and declined 2026-09-10, at one validation script: a subdirectory
would mean adding `[build-system]` to `pyproject.toml` so `uv sync` installs
the project editable. Worth doing, but not for one file — **revisit when a
second validation script is committed**, and name the directory `validation/`
rather than `tests/`. This is not a pytest suite: it audits a 1 TiB artifact,
needs the raw tree staged on GLADE, and runs for ~9 minutes, so anything that
collects `tests/` would pick up a file that cannot run off Derecho.

`finalize_gleam_zarr.py` (added 2026-09-10) is a **third** root-level entry
point, and it does not change that decision: the trigger stated above is a
second *validation* script, and this is not one. It sits at the root for the
same reason the other two do.

Whichever way that goes, do not paper over the import with `sys.path.insert`.

Test configs stay in `config/` beside the production one — they are inputs to
`gleam_zarr.py`, not to the verifier, and one home for all configs beats
splitting them by purpose.
