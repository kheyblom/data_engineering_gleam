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
| [config/config_zarr.yaml](config/config_zarr.yaml) | The config that drives it |
| [submit_gleam_zarr.sh](submit_gleam_zarr.sh) | Casper batch job wrapper |
| [utils/path_utils.py](utils/path_utils.py) | Config loading and path construction |
| [utils/zarr_utils.py](utils/zarr_utils.py) | Chunking, encoding, batched writes, resume checks |
| [utils/log_utils.py](utils/log_utils.py) | Logging to stdout and to the log file |
| [draft_zarr.ipynb](draft_zarr.ipynb) | Exploratory scratch work, not part of the pipeline |

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
with [submit_gleam_zarr.sh](submit_gleam_zarr.sh), which requests a Casper node
and runs the same command inside it:

```bash
./submit_gleam_zarr.sh                                  # default config
CONFIG=config/other.yaml ./submit_gleam_zarr.sh         # any other config
```

Run outside PBS, the script hands itself to `qsub` with the account taken from
`$PBS_ACCOUNT` — the same variable `qcmd` and `qinteractive` read, so exporting
it once in `~/.bashrc` covers everything. It has to be resolved this way rather
than written as a `#PBS -A` line, because directives are literal comments that
cannot expand a shell variable, and stock `qsub` reads only `PBS_DEFAULT` and
`PBS_DPREFIX` from the environment. Submitting by hand still works if you want a
different account: `qsub -A <PROJECT> submit_gleam_zarr.sh`.

The job asks for `1 node / 8 cpus / 32 GB / 12 h`; edit the `#PBS` lines to
change that. Before starting the build it checks
`num_workers` in the config against the `ncpus` the job was given and refuses to
start if the config asks for more, since an oversubscribed run would multiply
its chunks in flight straight past the memory it reserved. If a run hits
walltime, resubmit the same command — it resumes from the last commit.
Resubmitting after a completed run is a no-op.

Progress goes to both stdout and `<directories.logs>/<log_file>`; PBS job output
lands in `logs/` as well. Logs and all data outputs are gitignored.

## Configuration

Every setting lives in the YAML file passed to `--config`. The one shipped here
is [config/config_zarr.yaml](config/config_zarr.yaml).

### `directories`

| Key | Meaning |
| --- | --- |
| `download` | Root of the tree the download step wrote. The version directory, `raw/`, and the `zarr/` output directory all hang off this. |
| `logs` | Where the log file is written. Created if it does not exist. |

### `output_conventions`

| Key | Meaning |
| --- | --- |
| `filename` | Template for the store name. Any scalar at the top level of the config, plus any key under `output_conventions`, can be referenced by name; `{version}` is substituted in its on-disk form (`v_4_3_a`). Referring to a field the config does not define is an error. |
| `suffix` | A free-form tag fed to the template — `spatial` here — to distinguish stores built from the same source with different chunking or post-processing. |

### Top-level keys

| Key | Meaning |
| --- | --- |
| `log_file` | Name of the log file written inside `directories.logs`. Opened in append mode, so a resumed run adds to the existing log rather than truncating it. |
| `version` | GLEAM version, written as `v<major>.<minor><letter>` (e.g. `v4.3a`). Translated to the on-disk spelling `v_4_3_a` in exactly one place, `format_version`. A version that does not parse is an error. |
| `variables` | Either a list of variable names or the shorthand `all`. `all` expands to every variable directory actually present under `raw/<temporal_resolution>/`; an explicit list is kept in the order given, and a listed variable with no directory on disk is an error rather than a silent skip. |
| `temporal_resolution` | Which resolution to build — `daily` here. Selects the subdirectory under `raw/`, and only the one configured is built. |
| `grid_name` | Label for the grid the data is on (`native_0p1x0p1`). Used in the store name; it describes the data rather than reprojecting it, so changing it renames the output, it does not regrid. |
| `chunks` | Chunk size per dimension for the zarr store, in the dask convention where `-1` means the whole dimension. `time: 5, lat: -1, lon: -1` gives whole global maps chunked five timesteps deep — 124 MiB per chunk on the 1800x3600 grid. This is the main knob on both the shape of the output and the memory a run takes; see below. |

### The three settings that decide what a run costs

These last three keys are the ones worth understanding before launching a long
job. `timesteps_per_commit` sets how much work is at risk when a job dies;
`num_workers` and `file_cache_maxsize` set how much memory the run needs to stay
inside. Neither of the latter two changes the store that gets written — only the
resources used to write it.

#### `timesteps_per_commit` (default 100)

How many timesteps are written and committed to icechunk as one unit.

The dataset is assembled lazily and then pushed out batch by batch, with a
commit after each. That commit is the resume point: an interrupted run restarts
at the last committed batch, so this key is the trade-off between how much work
a crash costs you and how much commit overhead you pay. Smaller batches redo
less after a failure; larger batches spend proportionally less time committing.

Two things about it are easy to get wrong:

- **It is rounded, not used verbatim.** `commit_batch_size` rounds the value to
  the nearest whole multiple of the `time` chunk (never below one chunk). With
  `time: 5` and `timesteps_per_commit: 100`, the batch is exactly 100; with
  `time: 7` it would become 98. This is mandatory: xarray cannot append onto a
  partially filled chunk from dask, so every batch boundary must fall on a chunk
  boundary and only the *final* batch may be short. The `time` coordinate is
  additionally encoded with one chunk per batch for the same reason.
- **It is not a memory setting.** The `batch.nbytes` figure in the log — about
  34 GiB at the current settings — is the logical size of the batch, not what is
  resident. Each batch is streamed to the store chunk by chunk. Raising this
  value does not raise peak memory; the two keys below are what do.

Changing it between runs breaks resume. A store whose length is not a whole
number of the current batch size is rejected on resume, and the fix is to delete
and rebuild rather than patch around it.

#### `num_workers` (optional)

Size of the dask thread pool, and therefore how many chunks are in flight at
once.

This is the dominant term in peak memory: roughly `num_workers` x one chunk. At
`num_workers: 8` with 124 MiB chunks that is about 1 GiB of chunk buffers, on
top of the file caches below and the usual interpreter and library overhead.
Raising it buys parallelism in exact proportion to the memory it costs.

It **must not exceed the `ncpus` the batch job requested.** The config is
authoritative here — if you need more workers, raise `ncpus` in
[submit_gleam_zarr.sh](submit_gleam_zarr.sh) to match, rather than lowering the
config to fit the job. The submit script enforces this and refuses to launch on
a mismatch.

Leave it unset and dask sizes its own pool from the cpuset the job was given,
which is a reasonable default. It is applied by `configure_runtime` before
anything opens a file, since it does not take effect retroactively, and the
resolved value is logged — so a job killed for running out of memory can be read
back against the settings it actually ran with.

#### `file_cache_maxsize` (optional)

How many netCDF files xarray keeps open at once.

Every open netCDF file carries an HDF5 chunk cache of its own, 64 MiB by
default, so this is the second memory term and an easy one to overlook: the
files themselves are cheap, their caches are not. xarray's own default is 128
files, which here would reserve several gigabytes for caches that are mostly
idle.

The cache only has to cover the files a single batch actually touches. A batch
spans at most two year files per variable, so `32` is generous for the current
variable set and still an order of magnitude below what the default would hold.
Like `num_workers` it is applied by `configure_runtime` up front, falls back to
the library default when unset, and is logged as resolved.

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
