# Validating and tuning the GLEAM zarr build

Record of the testing run on **2026-09-09**, before the first production build
of the v4.3a daily store. Written down because most of what it found was not
what it went looking for.

The pipeline had never been run to completion — `logs/` was empty and no store
existed. The goal was to prove it worked end to end, and to choose
`timesteps_per_commit`, `num_workers` and `file_cache_maxsize` from
measurements rather than guesses, without paying for a full-scale failure.

**Outcome:** the pipeline was correct but unrunnable, and roughly 25x more
expensive than it needed to be. Three defects and one large cost lever were
found for **10.7 core-hours** of testing. The production run was launched at an
estimated **~57 core-hours**, against **~1830** for the same work on a `main`
node.

## Why it was done in tiers

A `main`-queue node on Derecho is allocated exclusively: a job there is billed
for all 128 cpus no matter how few it uses, so a blind 12 hour attempt costs
~1536 core-hours and teaches almost nothing if it dies. The `develop` queue
routes to `cpudev` with `place=pack:shared`, so it is billed only for the cpus
requested — which makes it possible to fail cheaply, repeatedly, on purpose.

Each tier was designed to fail as early and as cheaply as it could:

| Tier | What it did | Cost |
| --- | --- | --- |
| 0 | Config and path logic, no data read | free (login node) |
| 1 | Full build, value check, resume, rejection paths, on 2 variables x 1 year | 1.7 core-h |
| 2 | Open and merge all 644 real files, write nothing | included below |
| 3a | Write a throwaway store at full variable count for real timings | 9.0 core-h |

## Jobs run

| Job | Shape | Elapsed | Core-h | Result |
| --- | --- | --- | --- | --- |
| 7366310 | ncpus=4 | 0.011 h | 0.05 | **Segfault**, all 5 runs (finding 1) |
| 7366632 | ncpus=4 | 0.347 h | 1.39 | Passed except resume; a test-harness bug (finding 3) |
| 7367583 | ncpus=4 | 0.075 h | 0.30 | **All tier 1 checks pass**, 4.7x faster (finding 2) |
| 7367706 | ncpus=8 | 0.613 h | 4.91 | Tier 2 passed; tier 3a invalid, memory-starved (finding 4) |
| 7368172 | ncpus=8:mem=32GB | 0.505 h | 4.04 | Steady-state timings obtained (finding 5) |
| | | | **10.69** | |

## Findings

### 1. The pipeline could not read a single chunk on Python 3.14

Tier 1 died with exit 139 (SIGSEGV) on its first batch, five times out of five.
Bisected from the pipeline down to three lines of pure numpy:

```python
c = False
c |= (a == fill_value)      # right-hand side is a temporary
```

That is verbatim `xarray/coding/variables.py:130-132` — the CF fill-value
masking that every decoded read passes through. numpy's temporary-elision
optimisation misfires on Python 3.14, so it only crashes once the temporary
exceeds the elision threshold. That is why small HDF5-aligned reads survived
and full `lat`/`lon` slab reads did not.

- Every numpy from 2.3.4 to 2.5.3 crashes identically on Python 3.14.3.
- The **same** numpy 2.5.3 is clean on Python 3.13 and 3.12.

**Fix:** `requires-python = ">=3.13,<3.14"`. Do not raise that cap without
re-running `config/config_zarr_tiny.yaml`.

### 2. ~14x wasted decompression — the dominant cost

The raw files are HDF5-chunked `[12, 57, 113]`, **twelve timesteps deep**,
while `chunks: {time: 1}` reads one timestep at a time. Every compressed chunk
was therefore decompressed about twelve times. Measured on a real file:

| read pattern | time |
| --- | --- |
| 12 timesteps as one slab | 0.53 s |
| the same 12, one at a time | 7.32 s |

The fix does **not** require changing the output chunking, which is deliberate
(`time: 1` is what the `spatial` suffix means). Enlarging the HDF5 chunk cache
so it holds a whole twelve-deep row of chunks (1024 chunks, 302 MiB) lets the
twelve reads hit cache instead:

| | 12 single reads |
| --- | --- |
| default chunk cache | 9.06 s |
| 512 MiB cache | 0.87 s |

**10.4x**, and 8.24 s -> 1.09 s end to end through xarray with byte-identical
values. On the tier 1 build: **699 s -> 150 s (4.7x)**.

**Fix:** new `chunk_cache_size_mib` config key, applied in `configure_runtime`
via `netCDF4.set_chunk_cache` before anything opens a file.

### 3. Resume works; the test harness did not

Tier 1's resume check failed with `icechunk.ConflictError: expected parent ...
actual parent ...`. This was **not** a pipeline bug. `subprocess.run(timeout=)`
killed `uv`, but `uv run` spawns python as a child, and the orphan kept running
and committed the batch the resume was working on. The timeline is
unambiguous: the winning commit landed 188 s after the *orphan* started, not
30 s after the resume did.

The pipeline resumed from the right place, and icechunk's conflict detection
prevented the double write from corrupting the store. The harness now kills the
process group, which is what PBS does at walltime. Re-run: killed at timestep
300 on a batch boundary, resumed cleanly to 366.

### 4. The default 10 GB on a shared node silently starves the run

The first full-variable benchmark took **1928 s** for one batch. It was pinned
at its memory cap (`resources_used.mem = 10485884 kb` against a 10485760 kb
limit) with only ~2.2 of 8 cpus busy. With 14 variables open the chunk caches
want `14 x 512 MiB = 7 GiB`, and against 10 GB they thrash — putting the
finding-2 amplification straight back.

Re-run with `mem=32GB`: the same batch took **374 s**, a **5.2x** difference
from the memory request alone. Peak usage was 16.8 GB.

PBS grants a flat 10 GB on `cpudev` **regardless of ncpus** — asking for more
cpus does not buy more memory. Derecho bills on cpus only, so memory should be
requested freely.

### 5. This is a ~1 core job

Utilisation never rose above ~2 cores, whatever was requested:

| Job | ncpus | avg cpu | cores used |
| --- | --- | --- | --- |
| 7366632 | 4 | 52.0% | 2.08 (pre-fix, burning cpu on redundant decompression) |
| 7367706 | 8 | 26.9% | 2.15 (memory-starved) |
| 7367583 | 4 | 36.3% | 1.45 (healthy) |
| 7368172 | 8 | 15.8% | 1.26 (healthy) |
| production | 4 | — | 0.94 (healthy, 14 variables) |

The cause is structural: xarray's netCDF4 backend locks every read behind
`NETCDF4_PYTHON_LOCK`, a pair of process-global `SerializableLock` singletons,
and HDF5 decompression happens inside it. Extra dask threads have nothing to
do. Halving `num_workers` from 8 to 4 made the first batch *faster*
(374 s -> 234 s), which rules out any claim that workers were the constraint.

**Consequence:** queue choice was worth far more than any tuning parameter. On
Derecho only `develop` is shared; `main` -> `cpu` and `preempt` -> `pcpu` both
take exclusive whole nodes, so both bill 128 cpus for a job using one.

### 6. Smaller things worth keeping

- **`$NCPUS` lies on shared nodes.** PBS reported `NCPUS=1` whatever it
  granted, and `nproc` reports `OMP_NUM_THREADS`, which the job script pins to
  1. Only `len(os.sched_getaffinity(0))` is truthful. The submit script's
  oversubscription guard compared against `$NCPUS` and would have rejected
  every job.
- **`develop` caps walltime at 6 h** through a submission hook, not a PBS
  `resources_max`, so `qstat -Qf develop` shows no limit and a longer request
  is rejected only at submit time (free).
- **Chained resume jobs must use `afterany`, not `afterok`** — a job stopped by
  walltime exits non-zero, and that is exactly the case the next job resumes.
- **Startup is only 30.4 s** for all 644 file opens plus the merge, so chaining
  short jobs costs essentially nothing.
- **The store will be ~1.15 TB** on disk: the benchmark wrote 500 timesteps of
  169 GiB logical into 35 GB, a 4.9x compression ratio.
- **`chunks.time` was documented as 5** in both CLAUDE.md and the README while
  the config said 1. Corrected.

## What passed

Tier 1 (job 7367583), against a staged tree of 2 variables x 1 year:

| Check | Result |
| --- | --- |
| Build from nothing | 150 s, 4 batches |
| Values vs source netCDF | match at timesteps 0, 99, 100, 250, 365 |
| Calendar | 1980-01-01 .. 1980-12-31 through the pinned int64 encoding |
| Chunk shapes | data `(1, 1800, 3600)`, time `(100,)` |
| Fill value | 66.0% NaN — the ocean mask survives |
| Rerun a complete store | no-op, exit 0 |
| Kill and resume | killed at 300, resumed to 366 |
| Mismatched variable set | refused |

Tier 2 (all 644 real files, no write): `merge_variables`' exact time-axis check
passed across all 14 variables and 46 years, confirming **the download is
complete and consistent**. 16802 timesteps, 5.55 TiB logical, 169 batches, a
2-step final batch, 235,228 chunks; encoding covers every variable with no
stale netCDF encoding surviving.

## Changes made

| File | Change |
| --- | --- |
| `pyproject.toml` | `requires-python = ">=3.13,<3.14"` (finding 1) |
| `utils/zarr_utils.py` | `chunk_cache_size_mib` applied in `configure_runtime` (finding 2) |
| `config/config_zarr.yaml` | `chunk_cache_size_mib: 512`, `num_workers: 4` |
| `submit_gleam_zarr.sh` | `QUEUE`/`NCPUS`/`MEM`/`WALLTIME`/`AFTER` overrides; affinity-based guard (finding 6) |
| `config/config_zarr_tiny.yaml` | new — tier 1 config against a staged tree |
| `config/config_zarr_bench.yaml` | new — throwaway store for timing runs |
| `CLAUDE.md`, `README.md` | corrected chunk figures, documented the new key and the guard |

## Production configuration

```yaml
timesteps_per_commit: 100      # commit overhead invisible next to a ~300 s batch
num_workers: 4                 # ~3x more than the ~1 core actually used
file_cache_maxsize: 32         # a memory multiplier, not a speed knob
chunk_cache_size_mib: 512      # the one that mattered
```

```bash
QUEUE=develop NCPUS=4 MEM=32GB WALLTIME=06:00:00 ./submit_gleam_zarr.sh
# then chain resumes with AFTER=<previous jobid>
```

Launched 2026-09-09 as four chained 6 h jobs (24 h capacity against ~12-14 h of
work; a surplus job exits as a no-op in ~35 s). Steady state measured at
234-317 s per 100-timestep batch.

## Not done, and open questions

- **No `num_workers` sweep.** Finding 5 made it pointless — it would have been
  ~34 core-hours to draw a horizontal line. `ncpus=2` would very likely
  suffice; 4 was kept as cheap insurance.
- **No `timesteps_per_commit` sweep.** Commit overhead is invisible next to a
  ~300 s batch, and a larger batch only increases work lost to a kill.
- **No process-parallel spike.** Separate processes each get their own HDF5
  lock, so this is the only route past ~1 core; icechunk 2.2.0 already ships
  the machinery (`distributed.merge_sessions`, `dask.store_dask`,
  `to_icechunk(split_every=...)`). Deliberately skipped as unnecessary
  complexity once the queue choice made the run cheap. It is the thing to try
  if wallclock ever becomes the binding constraint.
- **`chunks: {time: 1}` is the worst possible layout for point time-series
  reads**, which touch all 16802 chunks of a variable. Correct for the
  `spatial` store; if that access pattern matters downstream it wants a second,
  time-chunked store rather than a change here.

## Reproducing the tests

The tier drivers were throwaway scripts, not committed. The two test configs
are, and both isolate their store by path, so neither can touch production:

```bash
# tier 1: stage 2 variables x 1 year as symlinks, then build to completion
uv run python gleam_zarr.py --config config/config_zarr_tiny.yaml

# tier 3a: real input, throwaway store under a 'bench' suffix
uv run python gleam_zarr.py --config config/config_zarr_bench.yaml
```

Per-batch timings can be read straight out of the log — the pipeline already
logs a timestamped line when a batch starts writing and another when it
commits. Actual core utilisation comes from `qhist -u $USER -c` (the `Avg CPU`
column is percent of allocated cores).
