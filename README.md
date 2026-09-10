# data_engineering_gleam

Convert the raw GLEAM netCDF files (evaporation, soil moisture, and friends)
into a single [icechunk](https://icechunk.io) backed zarr store on NCAR
Derecho/Casper.

The pipeline is one script driven by one YAML config. Nothing is hardcoded:
paths, version, variables, chunk sizes, and the execution settings all come from
the config, so a different store is a different config file rather than a code
change.

The write is **incremental and restartable** — timesteps are pushed out in
batches, each committed to the icechunk repository before the next starts. A run
killed by walltime leaves a valid store committed up to its last batch, and
relaunching with the same config picks up from there.

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
```

Repository files:

| Path | What it is |
| --- | --- |
| [gleam_zarr.py](gleam_zarr.py) | The pipeline: open, merge, chunk, write, resume |
| [verify_gleam_zarr.py](verify_gleam_zarr.py) | Audits a finished store against the raw files. Read only throughout |
| [finalize_gleam_zarr.py](finalize_gleam_zarr.py) | Writes attributes, tags a snapshot, collects unreachable objects. The only script here that mutates a finished store |
| [config/config_zarr.yaml](config/config_zarr.yaml) | The config that drives it |
| [submit_gleam_zarr.sh](submit_gleam_zarr.sh) | Derecho batch job wrapper |
| [utils/path_utils.py](utils/path_utils.py) | Config loading and path construction |
| [utils/zarr_utils.py](utils/zarr_utils.py) | Chunking, encoding, batched writes, resume checks |
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

Once a store has been verified, [finalize_gleam_zarr.py](finalize_gleam_zarr.py)
does the three things that turn it into a deliverable. It takes exactly one
action per run and writes nothing without `--apply`:

```bash
# write the config's attrs section plus the derived coverage and grid attributes
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --attrs --apply

# name the current branch tip; tags are immutable
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml \
    --tag v4.3a-verified-20260910 --apply

# delete objects no surviving snapshot references
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --gc --apply
```

`--gc` is the irreversible one. Every write through `to_icechunk` forks the
session, and the fork's snapshot is not part of the branch's ancestry, so a
store accumulates one unreachable snapshot per commit as a matter of course —
plus the chunks of any batch killed before it could commit. Collection removes
only what nothing references, and the script checks that claim by comparing
reachable bytes and history length either side of the call. Run
`verify_gleam_zarr.py --phases structure,sweep` afterwards regardless: the sweep
is the only full-coverage check there is.

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
| `chunks` | Chunk size per dimension for the zarr store, in the dask convention where `-1` means the whole dimension. `time: 1, lat: -1, lon: -1` gives one whole global map per chunk — 24.7 MiB on the 1800x3600 grid — which is what the `spatial` suffix in the store name refers to: it is the layout for reading maps, and the worst one for reading a long time series at a point. This is the main knob on both the shape of the output and the memory a run takes; see below. |

### The four settings that decide what a run costs

These are the keys worth setting deliberately before a long job.
`timesteps_per_commit` decides how much work a crash throws away; the other
three decide how fast the run goes and how much memory it needs. None of the
latter three change the store that gets written, only the resources used to
write it — and `chunk_cache_size_mib` is worth more than the other two put
together.

| Key | Default | What it controls |
| --- | --- | --- |
| `timesteps_per_commit` | 100 | Timesteps written and committed to icechunk as one unit, and so the point a killed run resumes from. Smaller batches redo less after a failure; larger ones spend proportionally less time committing. Two traps: it is **rounded** to a whole multiple of the `time` chunk (`time: 7` turns 100 into 98), because xarray cannot append onto a partially filled chunk from dask — at the configured `time: 1` the rounding is a no-op, so this only bites if `time` is raised; and it is **not** a memory knob — the ~34 GiB `batch.nbytes` in the log is the batch's logical size, not what is resident, since batches stream chunk by chunk. Changing it between runs breaks resume: a store whose length is not a whole number of the current batch size is rejected, and the fix is to rebuild. |
| `num_workers` | dask's own, sized from the cpuset | Size of the dask thread pool, and so how many chunks are in flight — roughly `num_workers` x one chunk, about 200 MiB at 8 workers and 24.7 MiB chunks — small enough at this chunk size that the open file cache below is the larger memory term. Note that it buys much less parallelism than it looks like it should: xarray locks every netCDF read behind one process-global HDF5 lock, so reads serialize no matter how many workers there are, and only the zarr compression on the write side scales. It **must not exceed the job's `ncpus`**: if you need more workers, raise `ncpus` in [submit_gleam_zarr.sh](submit_gleam_zarr.sh) rather than lowering the config, which the script checks and refuses to launch on. |
| `chunk_cache_size_mib` | HDF5's own, ~16 MiB | Decompressed HDF5 chunk cache held per open netCDF file. **The single largest performance setting here.** The raw files are chunked `[12, 57, 113]` — twelve timesteps deep — while the store is written one timestep at a time, so unless this holds a whole twelve-deep row of chunks (1024 chunks, 302 MiB) HDF5 decompresses every chunk twelve times over. Measured on real data: reading twelve timesteps one at a time takes 9.06 s at the default and 0.87 s at 512 MiB, and 8.24 s vs 1.09 s end to end through xarray. It changes nothing about the store. Budget it against `file_cache_maxsize`, which multiplies it, and against `num_workers`, since threads reading distant timesteps touch different chunk rows. |
| `file_cache_maxsize` | xarray's, 128 files | How many netCDF files stay open, each holding its own 64 MiB HDF5 chunk cache — the second memory term, and easy to overlook because the files are cheap but their caches are not. It only has to cover the files one batch touches, at most two year files per variable, so `32` is generous here and well below the several idle GiB the default would hold. |

Both memory settings are applied by `configure_runtime` before anything opens a
file, since neither takes effect retroactively, and the resolved values are
logged — so a job killed for running out of memory can be read back against the
settings it actually ran with.

## How the build works

1. **Resolve** — expand `variables: all` against the directories present under
   `raw/<temporal_resolution>/`, and list each variable's yearly files in time
   order.
2. **Open** — each variable is opened with `open_mfdataset` across its whole
   year range, concatenated along time in the order given. Note that
   `parallel=True` is deliberately omitted: opening netCDF from several threads
   crashes the HDF5 library in this build. Do not add it.
3. **Merge** — variables are merged with `join='exact'` after an explicit check
   that they share an identical time axis. A mismatch means an incomplete
   download and is raised as an error; without the check it would surface much
   later as a silently NaN-filled variable.
4. **Chunk** — `-1` entries in `chunks` are resolved against the now-known
   dimension sizes. `open_mfdataset` chunks per file, so time chunks follow the
   yearly file boundaries until this explicit rechunk squares them up.
5. **Encode** — the netCDF encoding xarray carries over (`zlib`, `chunksizes`,
   ...) is rejected by the zarr backend, so `build_encoding` clears each
   variable's `.encoding` and rebuilds it rather than overriding keys. Time is
   pinned to `days since 1900-01-01`, proleptic_gregorian, int64 so that every
   appended batch encodes identically.
6. **Write** — the first batch lays down the arrays with that encoding; every
   later batch appends along `time`. Each is committed before the next starts.

On resume, `check_resume` compares the stored variables and the overlapping
timesteps against the incoming dataset and refuses to append onto a store built
from a different configuration.

## Conventions

Single quotes, f-strings for log messages, Google-style docstrings with
Args / Returns / Raises on every non-trivial function. Comments explain *why* a
choice was made — the HDF5 threading note, the chunk-boundary rule — not what
the line does. Module docstrings carry the context needed to read the file.

There is no test suite, linter, or CI configured.
