# data_engineering_gleam

Convert the raw GLEAM netCDF files (evaporation, soil moisture, and friends)
into a single [icechunk](https://icechunk.io) backed zarr store on NCAR
Derecho/Casper.

The pipeline is one script driven by one YAML config. Nothing is hardcoded:
paths, version, variables, chunk sizes, and the execution settings all come from
the config, so a different store is a different config file rather than a code
change.

The write is **incremental and restartable**: the store goes out in pieces, each
committed to the icechunk repository before the next starts, so a run killed by
walltime leaves a valid store and relaunching with the same config picks up from
there. What a piece is follows the chunking, and `write_strategy` picks between
the two — `append` writes a batch of timesteps at a time, `region` writes one
lat/lon block of the whole record at a time. See
[How the build works](#how-the-build-works).

Two stores are built from the same source files, differing only in chunking:
`spatial` for reading maps, `temporal` for reading time series. Neither is
derived from the other, so each is verified against the raw files on its own.

## Layout

The download step (not part of this repo) writes the raw tree; this repo writes
the `zarr` directory next to it:

```
<directories.download>/<version>/raw/<temporal_resolution>/<variable>/*.nc
<directories.download>/<version>/zarr/<output_conventions.filename>
```

The version is spelled `v4.3a` in the config but `v_4_3_a` on disk, and the
store name carries the same directory spelling. With the config shipped here
that resolves to:

```
/glade/derecho/scratch/$USER/data/gleam/v_4_3_a/raw/daily/<variable>/*.nc
/glade/derecho/scratch/$USER/data/gleam/v_4_3_a/zarr/gleam.v_4_3_a.daily.native_0p1x0p1.spatial.zarr
/glade/derecho/scratch/$USER/data/gleam/v_4_3_a/zarr/gleam.v_4_3_a.daily.native_0p1x0p1.temporal.zarr
```

The `suffix` field is the only difference between those two names, and the only
difference between the configs that build them is `chunks`, `write_strategy` and
the block size.

Repository files:

| Path | What it is |
| --- | --- |
| [gleam_zarr.py](gleam_zarr.py) | The pipeline: open, merge, chunk, write, resume |
| [verify_gleam_zarr.py](verify_gleam_zarr.py) | Audits a finished store against the raw files. Read only throughout |
| [finalize_gleam_zarr.py](finalize_gleam_zarr.py) | Writes attributes, tags a snapshot, collects unreachable objects. The only script here that mutates a finished store |
| [config/config_zarr.yaml](config/config_zarr.yaml) | The config that drives it |
| [submit_gleam_zarr.sh](submit_gleam_zarr.sh) | Derecho batch job wrapper |
| [utils/path_utils.py](utils/path_utils.py) | Config loading and path construction |
| [utils/zarr_utils.py](utils/zarr_utils.py) | Chunking, encoding, both write strategies, resume checks |
| [utils/log_utils.py](utils/log_utils.py) | Logging to stdout and to the log file |
| [draft_zarr.ipynb](draft_zarr.ipynb) | Exploratory scratch work, not part of the pipeline |
| [TESTING.md](TESTING.md) | How the pipeline was validated and tuned, and what that found |

## Where the finished store lives

The v4.3a daily store is built, verified and finalized:

```
/glade/derecho/scratch/kheyblom/data/gleam/v_4_3_a/zarr/gleam.v_4_3_a.daily.native_0p1x0p1.spatial.zarr
```

16802 timesteps, 14 variables, 1980-01-01 .. 2025-12-31 on the 1800x3600 grid;
1.032 TiB compressed, 0.186 of logical. Verified against the source netCDF files
on 2026-09-10 and tagged **`v4.3a-verified-20260910`**.

Read it through the tag rather than the branch. A tag in icechunk is immutable,
so it cannot be moved by a later commit, and `consolidated=False` is required —
icechunk's manifest does the job zarr's consolidated metadata would:

```python
import icechunk
import xarray as xr

path = ('/glade/derecho/scratch/kheyblom/data/gleam/v_4_3_a/zarr/'
        'gleam.v_4_3_a.daily.native_0p1x0p1.spatial.zarr')
repository = icechunk.Repository.open(icechunk.local_filesystem_storage(path))
session = repository.readonly_session(tag='v4.3a-verified-20260910')
dataset = xr.open_zarr(session.store, consolidated=False)
```

The store describes itself: `dataset.attrs` carries the provenance, the coverage
and grid extents, a `chunking` note on what this layout is and is not good for,
and `known_data_gaps` recording that `E` is absent on 25 upstream-missing days.

**It is on scratch, which is purged.** That is a deliberate, accepted risk —
moving it to `/glade/campaign/univ/umic0112` is a separate piece of work. Until
then the store should be treated as reproducible rather than archived: rebuilding
it costs ~57 core-hours, and the raw tree it is built from (~1.7 TiB) sits on the
same purge-prone filesystem.

## Finalizing a store

Turning a verified store into a deliverable is a fixed procedure, run by
[finalize_gleam_zarr.py](finalize_gleam_zarr.py). It takes exactly one action
per invocation and writes nothing without `--apply`, so every step can be
previewed before it happens.

**The order is load-bearing**, which is the one thing to get right:

1. `--attrs` first, so the snapshot the tag names already carries the
   provenance.
2. `--tag` second. Tags are immutable and cannot be moved, so a tag created
   before the attributes exist permanently names a store that does not describe
   itself.
3. `--gc` last. A tag is a ref, so tagging first makes the reachability set
   explicit rather than leaving it implicit in wherever `main` happens to point.

`--status` reports where a store already stands, including which of the three
steps remain, so the order does not have to be remembered:

```bash
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --status
```

### The procedure

```bash
# 0. pre-flight. Only one writer can hold the icechunk branch, so no build or
#    resume job may be running against this store
qstat -u $USER

# 1. baseline, and the state to compare everything against afterwards
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --status

# 2. attributes. Preview, then apply
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --attrs
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --attrs --apply

# 3. tag. Convention is <version>-verified-<YYYYMMDD>, dated so a later
#    re-verification can add its own tag without ambiguity
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml \
    --tag v4.3a-verified-20260910 --apply

# 4. collect unreachable objects. Read the dry run before applying
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --gc
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --gc --apply

# 5. gates
uv run python verify_gleam_zarr.py --config config/config_zarr.yaml \
    --phases structure,sweep                       # ~35 s, full chunk coverage
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --gc --apply
                                                   # must report all zeros
uv run python verify_gleam_zarr.py --config config/config_zarr.yaml
                                                   # ~9 min, the final word
```

All of it runs on a login node for free. Step 4 is the only irreversible one.

### Reading a garbage collection summary

**A count of unreachable objects means nothing on its own.** "171 snapshots"
reads equally as *the entire commit history* and as *171 objects nothing points
at*, and this repo once recorded the wrong one of those and deferred a safe
cleanup for a day over it (TESTING.md, finding 9). It only becomes meaningful
against the reachable history and the objects actually on disk, which is what
`--status` prints together and why it is step 1.

For the record, unreachable snapshots are **expected and normal**: every write
through `to_icechunk` forks the session, and the fork's snapshot never joins the
branch ancestry, so a store accumulates one per commit as a matter of course.
Orphaned *chunks*, by contrast, come from batches killed before they could
commit. So the snapshot count scales with the length of the run and the chunk
count does not — 169 batches plus 2 killed gave exactly 171 fork snapshots and
2012 orphaned chunks.

`garbage_collect` deletes an object only if no surviving snapshot references it.
`expire_snapshots` is **not** part of this procedure: it reclaims essentially no
bytes, since the branch tip already references every live chunk, and all it
would do is discard the build's history.

### If a gate fails

`--gc --apply` compares reachable bytes and history length either side of the
collection and exits non-zero if either moved, since collection is defined not
to change them. If that trips, or if the `sweep` phase reports a missing chunk
that raw says should hold data, **the store is damaged and there is no undo** —
garbage collection cannot be reversed. Recovery is a restore from another copy
if one exists, and a rebuild (~57 core-hours) if one does not. This is the
reason to know whether a second copy exists *before* running step 4.

## Running it

Create or refresh the environment (there is no package install step — the script
is run from the repo root):

```bash
uv sync
```

Run the build directly, which is fine for a short test or a small subset:

```bash
uv run python gleam_zarr.py --config config/config_zarr.yaml
```

A full build is far too heavy for a login node, so **submit it as a batch job**
with [submit_gleam_zarr.sh](submit_gleam_zarr.sh), which requests a Derecho node
and runs the same command inside it:

```bash
./submit_gleam_zarr.sh                                  # default config
CONFIG=config/other.yaml ./submit_gleam_zarr.sh         # any other config

# a short test on the shared develop queue, billed for the cpus it asks for
QUEUE=develop NCPUS=8 WALLTIME=00:30:00 \
    CONFIG=config/config_zarr_tiny.yaml ./submit_gleam_zarr.sh

# the region path needs memory rather than cpus: one block is held whole, and
# a shared develop job gets a flat 10 GB default whatever ncpus it asked for
QUEUE=develop NCPUS=2 MEM=96GB WALLTIME=06:00:00 ./submit_gleam_zarr.sh

# chain a resume behind a running job so the two never write the store at once
AFTER=<jobid> ./submit_gleam_zarr.sh
```

Run outside PBS, the script hands itself to `qsub` with the account taken from
`$PBS_ACCOUNT` — the same variable `qcmd` and `qinteractive` read, so exporting
it once in `~/.bashrc` covers everything. It has to be resolved this way rather
than written as a `#PBS -A` line, because directives are literal comments that
cannot expand a shell variable, and stock `qsub` reads only `PBS_DEFAULT` and
`PBS_DPREFIX` from the environment. Submitting by hand still works if you want a
different account: `qsub -A <PROJECT> submit_gleam_zarr.sh`.

The queue shape defaults to the production run — `main`, one whole node of 128
cpus, 12 hours, Derecho's ceiling — and is overridden with the `QUEUE`, `NCPUS`
and `WALLTIME` variables above rather than by editing `#PBS` lines, which are
literal comments that cannot expand a variable. `job_priority` is added only on
`main`, since the develop queue rejects it. Note that `main` allocates whole
nodes exclusively, so a job there is billed for all 128 cpus whatever it uses,
while `develop` is shared and bills only what it requests.

Before starting the build the script checks `num_workers` in the config against
the cpus the job can actually use and refuses to start if the config asks for
more, since an oversubscribed run would multiply its chunks in flight straight
past the memory it reserved. That count comes from the process affinity mask,
because neither `$NCPUS` nor `nproc` is trustworthy here: on a shared develop
node PBS reports `NCPUS=1` whatever it granted, and `nproc` reports
`OMP_NUM_THREADS`, which the script pins to 1.

If a run hits walltime, resubmit the same command — it resumes from the last
commit. Resubmitting after a completed run is a no-op.

Progress goes to both stdout and `<directories.logs>/<log_file>`; PBS job output
lands in `logs/` as well. Logs and all data outputs are gitignored.

## Configuration

Every setting lives in the YAML file passed to `--config`. The one shipped here
is [config/config_zarr.yaml](config/config_zarr.yaml).

### `directories`

| Key | Meaning |
| --- | --- |
| `download` | Root of the tree the download step wrote. The version directory, `raw/`, and the `zarr/` output directory all hang off this. |
| `raw` | Optional. Root of the tree holding the raw netCDF files, when they are not under `download`. Like `download` it is the root *above* the version directory, not the `raw/` segment itself. Unset, it falls back to `download`, which is what every config here does. It exists so a store that has been moved can still be verified against raw files that did not move with it — `store_path` follows `download`, so without this the inputs and outputs are pinned to the same filesystem. |
| `logs` | Where the log file is written. Created if it does not exist. |

### `output_conventions`

| Key | Meaning |
| --- | --- |
| `filename` | Template for the store name. Any scalar at the top level of the config, plus any key under `output_conventions`, can be referenced by name; `{version}` is substituted in its on-disk form (`v_4_3_a`). Referring to a field the config does not define is an error. |
| `suffix` | A free-form tag fed to the template — `spatial` here — to distinguish stores built from the same source with different chunking or post-processing. |

### `attrs`

Optional. The store's global attributes, written by `gleam_zarr.py` on the first
batch and by `finalize_gleam_zarr.py --attrs` onto a store that already exists.
Each string value is a template over the same fields the `filename` template
uses, with `{version_label}` for the version as the config writes it (`v4.3a`)
alongside `{version}` for the on-disk spelling (`v_4_3_a`).

Three things stay out of this section deliberately. The coverage and grid
attributes (`time_coverage_*`, `geospatial_*`) are derived from the data by
`derive_attrs`, so they cannot drift from what was actually written. The source
files' own attributes are carried over by the merge and kept underneath these,
so GLEAM's upstream provenance survives. And `Conventions` is `ACDD-1.3` rather
than a CF version: the variables' `standard_name` and `units` come through
unaltered from upstream and are not CF-valid, which the `cf_compliance`
attribute states outright rather than leaving a consumer to discover.

### Top-level keys

| Key | Meaning |
| --- | --- |
| `log_file` | Name of the log file written inside `directories.logs`. Opened in append mode, so a resumed run adds to the existing log rather than truncating it. |
| `version` | GLEAM version, written as `v<major>.<minor><letter>` (e.g. `v4.3a`). Translated to the on-disk spelling `v_4_3_a` in exactly one place, `format_version`. A version that does not parse is an error. |
| `variables` | Either a list of variable names or the shorthand `all`. `all` expands to every variable directory actually present under `raw/<temporal_resolution>/`; an explicit list is kept in the order given, and a listed variable with no directory on disk is an error rather than a silent skip. |
| `temporal_resolution` | Which resolution to build — `daily` here. Selects the subdirectory under `raw/`, and only the one configured is built. |
| `grid_name` | Label for the grid the data is on (`native_0p1x0p1`). Used in the store name; it describes the data rather than reprojecting it, so changing it renames the output, it does not regrid. |
| `chunks` | Chunk size per dimension for the zarr store, in the dask convention where `-1` means the whole dimension. This is the main knob on both the shape of the output and what a run costs. `time: 1, lat: -1, lon: -1` gives one whole global map per chunk, 24.7 MiB on the 1800x3600 grid — the `spatial` layout, built for reading maps and the worst possible one for a point time series, which touches all 16802 chunks of a variable. `time: -1, lat: 20, lon: 20` gives one 2x2 degree tile through the whole record, 25.6 MiB — the `temporal` layout, where that time series is a single chunk and a global map is the worst case instead. |
| `write_strategy` | `append` (the default) or `region`; how the store is filled. It is not a free choice: it has to match the chunking, and `resolve_write_strategy` raises rather than letting a mismatch through, because both mismatches fail quietly. An `append` build of a store whose time chunk spans the record collapses into a single uninterruptible commit that a walltime kill loses entirely; a `region` build of a shallowly chunked one reads the whole record into memory to write chunks one timestep deep. |
| `block_shape` | `region` only. The lat/lon block read and committed as one unit, which must be a whole number of output chunks. Two things decide it. It has to be much wider than a chunk, because the raw files are chunked `[12, 57, 113]` and a block only 20 cells wide would decompress each 57x113 source chunk once for every tile it covers. And **it has to span `lon` entirely** (`lon: -1`): a narrower block reads a strided subset of the source chunks — 11 of every 32 along lon, scattered through the file — rather than contiguous runs, which measured pathologically slow on dense variables at identical volume. At `lat: 200, lon: -1` that is 45.1 GiB resident per block against 1.43x read amplification, a trade worth making because Derecho bills cpus and not memory. It is also the resume granularity, so it is what a killed job redoes. |

### The four settings that decide what a run costs

These are the keys worth setting deliberately before a long job.
`timesteps_per_commit` (on the `append` path) and `block_shape` (on the
`region` path) decide how much work a crash throws away; the other three decide
how fast the run goes and how much memory it needs. None of the latter three
change the store that gets written, only the resources used to write it.

Note that `chunk_cache_size_mib` is worth far more than the other two on the
`append` path and much less on the `region` path, for the same reason: it exists
to stop a twelve-deep source chunk being decompressed once per timestep, and the
`region` path reads each source chunk once anyway, in one hyperslab per file.

| Key | Default | What it controls |
| --- | --- | --- |
| `timesteps_per_commit` | 100 | `append` only. Timesteps written and committed to icechunk as one unit, and so the point a killed run resumes from. Smaller batches redo less after a failure; larger ones spend proportionally less time committing. Two traps: it is **rounded** to a whole multiple of the `time` chunk (`time: 7` turns 100 into 98), because xarray cannot append onto a partially filled chunk from dask — at the configured `time: 1` the rounding is a no-op, so this only bites if `time` is raised; and it is **not** a memory knob — the ~34 GiB `batch.nbytes` in the log is the batch's logical size, not what is resident, since batches stream chunk by chunk. Changing it between runs breaks resume: a store whose length is not a whole number of the current batch size is rejected, and the fix is to rebuild. |
| `num_workers` | dask's own, sized from the cpuset | Size of the dask thread pool, and so how many chunks are in flight — roughly `num_workers` x one chunk, about 200 MiB at 8 workers and 24.7 MiB chunks — small enough at this chunk size that the open file cache below is the larger memory term. Note that it buys much less parallelism than it looks like it should: xarray locks every netCDF read behind one process-global HDF5 lock, so reads serialize no matter how many workers there are, and only the zarr compression on the write side scales. It **must not exceed the job's `ncpus`**: if you need more workers, raise `ncpus` in [submit_gleam_zarr.sh](submit_gleam_zarr.sh) rather than lowering the config, which the script checks and refuses to launch on. |
| `chunk_cache_size_mib` | HDF5's own, ~16 MiB | Decompressed HDF5 chunk cache held per open netCDF file. **The single largest performance setting on the `append` path.** The raw files are chunked `[12, 57, 113]` — twelve timesteps deep — while an `append` build writes one timestep at a time, so unless this holds a whole twelve-deep row of chunks (1024 chunks, 302 MiB) HDF5 decompresses every chunk twelve times over. Measured on real data: reading twelve timesteps one at a time takes 9.06 s at the default and 0.87 s at 512 MiB, and 8.24 s vs 1.09 s end to end through xarray. On the `region` path it is worth much less, because a block is one contiguous hyperslab per file and every source chunk it touches is decompressed once regardless. It changes nothing about the store. Budget it against `file_cache_maxsize`, which multiplies it. |
| `file_cache_maxsize` | xarray's, 128 files | How many netCDF files stay open, each holding its own HDF5 chunk cache — the second memory term, and easy to overlook because the files are cheap but their caches are not. It only has to cover the files one unit of work touches: at most two year files per variable on the `append` path, so `32`; every file of one variable on the `region` path, which is variable-major, so `64`. |

Both memory settings are applied by `configure_runtime` before anything opens a
file, since neither takes effect retroactively, and the resolved values are
logged — so a job killed for running out of memory can be read back against the
settings it actually ran with.

## How the build works

1. **Resolve** — expand `variables: all` against the directories present under
   `raw/<temporal_resolution>/`, and list each variable's yearly files in time
   order.
2. **Open** — each variable is opened with `open_mfdataset` across its whole
   year range, concatenated along time in the order given. The chunks a file is
   opened with are the unit dask reads in, which is *not* the same thing as the
   store's chunking: on the `region` path it is the block, because opening 644
   files as 20x20 tiles would build some ten million dask chunks before a byte
   is read. Note that `parallel=True` is deliberately omitted: opening netCDF
   from several threads crashes the HDF5 library in this build. Do not add it.
3. **Merge** — variables are merged with `join='exact'` after an explicit check
   that they share an identical time axis. A mismatch means an incomplete
   download and is raised as an error; without the check it would surface much
   later as a silently NaN-filled variable.
4. **Chunk** — `-1` entries in `chunks` are resolved against the now-known
   dimension sizes, which is also when `write_strategy` can be checked against
   them. On the `append` path `open_mfdataset` chunks per file, so time chunks
   follow the yearly file boundaries until an explicit rechunk squares them up;
   the `region` path writes from memory rather than from dask and leaves the
   blocks chunked the way they were read.
5. **Encode** — the netCDF encoding xarray carries over (`zlib`, `chunksizes`,
   ...) is rejected by the zarr backend, so `build_encoding` clears each
   variable's `.encoding` and rebuilds it rather than overriding keys. Time is
   pinned to `days since 1900-01-01`, proleptic_gregorian, int64 so that every
   appended batch encodes identically. Index coordinates are kept as a single
   chunk rather than inheriting the chunking of the data they index, which at
   `lat: 20` would otherwise split the 1800-long `lat` coordinate into 90.
6. **Write**, one of two ways:
   - `append` — the first batch lays down the arrays with that encoding; every
     later batch appends along `time`. Each is committed before the next starts.
   - `region` — `create_skeleton` lays down the group, the array metadata and
     the coordinates with `compute=False`, which writes no data chunk at all,
     and commits. Each block is then read into memory whole, written into the
     skeleton with `region=`, and committed on its own.

On resume the `append` path's `check_resume` compares the stored variables and
the overlapping timesteps against the incoming dataset, and refuses to append
onto a store built from a different configuration. The `region` path cannot lean
on the store's length, since every array is full-sized from the moment the
skeleton exists, so `check_resume_region` compares the variables, all three
coordinates and the chunk grid instead, and `committed_blocks` reads the blocks
already written back out of the commit messages. Progress lives in the history
rather than in the store so that resuming needs nothing but the ancestry
icechunk keeps anyway, and so `finalize_gleam_zarr.py` does not have to strip
build bookkeeping out of the attributes it publishes.

## Conventions

Single quotes, f-strings for log messages, Google-style docstrings with
Args / Returns / Raises on every non-trivial function. Comments explain *why* a
choice was made — the HDF5 threading note, the chunk-boundary rule — not what
the line does. Module docstrings carry the context needed to read the file.

There is no test suite, linter, or CI configured.
