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

## Production run outcome (2026-09-09 14:11 -> 2026-09-10 04:24)

The chain ran as designed. Two jobs hit walltime and handed off; the third
finished the store; the spare no-opped.

| Job | Elapsed | Core-h | Ended | Wrote |
| --- | --- | --- | --- | --- |
| 7368794 | 6.02 h | 24.06 | walltime | timesteps 0 -> 8200 |
| 7368795 | 6.02 h | 24.10 | walltime | 8200 -> 15300 |
| 7368796 | 2.19 h | 8.76 | exit 0 | 15300 -> 16802, complete |
| 7368797 | 36 s | 0.04 | exit 0 | no-op, `store is already complete` |
| | **14.2 h** | **57.0** | | 169 batches |

**Estimates against reality**, from four bench batches:

| | estimated | actual |
| --- | --- | --- |
| Core-hours | ~57 | **57.0** |
| Store on disk | ~1.15 TB | **1.041 TiB** |
| Wallclock | ~12.5 h | 14.2 h |
| Batch rate | 234-317 s | 267 s across 158 of 169 batches |

Wallclock ran ~14% over because **11 batches were slow**, all of them clustered
just after a resume (8400-9700 and 15300-16000, peaking at 1511 s against a
267 s norm). They are scattered rather than consecutive, which points at
neighbour contention on the shared `develop` nodes rather than anything in the
pipeline — the batches after each cluster return to 267 s, and the final bucket
of the run is normal. **Budget ~15-20% wallclock margin on shared nodes.** The
4-job chain absorbed it with no intervention.

Note this also means the per-job rates (0.13 / 0.11 / 0.06 GiB/s for jobs 1-3)
are *not* progressive degradation, which is how they first read. They are an
artefact of where the job boundaries fell relative to the two slow clusters.

### Verification of the finished store

| Check | Result |
| --- | --- |
| Timesteps committed | 16802 |
| Variables | 14 |
| Time range | 1980-01-01 .. 2025-12-31 |
| Time axis | monotonic, no gaps or duplicates |
| Chunks | `(1, 1800, 3600)` |
| Values vs source netCDF | `E` and `SMs` match at t = 0, 8199, 8200, 15299, 15300, 16801 |

The value checks deliberately straddle **both resume seams** (t=8199/8200 and
t=15299/15300), the only timesteps in the store whose neighbouring batches were
written by different processes. Both are clean.

### Storage, and the orphaned chunks

- On disk: **1.041 TiB** (1,144,508,395,181 bytes)
- Referenced by live snapshots: **1.032 TiB** (`chunk_storage_stats`)
- Orphaned: **8.57 GiB / 2012 chunks**, measured with
  `garbage_collect(..., dry_run=True)`

2012 orphaned chunks is ~143 timesteps x 14 variables, which is exactly the two
partial batches that were in flight when jobs 1 and 2 were killed. Their
sessions never committed, so nothing references them. That is 0.8% of the store
and was **not** cleaned up — see the reasoning under open questions.

## Full verification of the store (2026-09-10)

The check above was written to answer "did the chain hand off cleanly", and it
did that: 12 planes across 2 variables, straddling both resume seams. It is not
enough to sign the store off as a deliverable. 12 planes is 0.005% of it, two
variables leave twelve unexamined, and nothing had looked at whether every
chunk the store claims to hold actually exists.

This pass was run on a login node for **free** — **103 checks, 0 failures, in
8.7 minutes** on one core, almost all of it waiting on raw netCDF
decompression. It is committed as `verify_gleam_zarr.py`, driven by the same
config as the build.

**Outcome:** no defect in the store. Everything that first read as a failure
turned out to be either a property of the upstream GLEAM data or a wrong
assertion on my part, which is the finding worth writing down — three of the
six phases as first written produced false alarms, and a verification script
that cries wolf is worse than none.

### What was checked

| Phase | Cost | Coverage |
| --- | --- | --- |
| `structure` | metadata only | dims, chunk shapes, dtypes, fill values, codecs, CF and global attributes, encoding, snapshot count |
| `index` | 644 metadata reads | all 14 raw time axes agree; store time/lat/lon bit-identical to raw |
| `samples` | 144 planes | random stratified + all 45 file seams + every batch and resume boundary |
| `sweep` | metadata only, 32 s | **all 235,228 chunks** — presence, placement, compressed size |
| `identities` | 4 days | GLEAM's own component sums |
| `ranges` | 84 planes | physical bounds from each variable's `units`, plus one raw plane per variable whose footprint moves |

The raw side is read with `netCDF4` and `set_auto_mask(False)`, so the fill
sentinel arrives as the literal -999. Comparing against `xr.open_dataset`
instead would push both sides through the same CF decoding path and could only
prove the pipeline agrees with itself — not that -999 became NaN. Comparisons
are exact, never `allclose`: the pipeline does no arithmetic on the values, so
anything short of bit-identical is a defect.

| Check | Result |
| --- | --- |
| Dimensions | `time=16802, lat=1800, lon=3600`, 14 variables |
| Calendar | 1980-01-01 .. 2025-12-31, one timestep per day of the span, monotonic, no duplicates — 46 complete years including all 12 leap days |
| Time values | bit-identical to the raw `days since 1900` int64s, through the pinned encoding |
| Time axis agreement | identical across all 14 variables, so the merge reindexed nothing |
| `lat` / `lon` | bit-identical to raw |
| Chunk shapes | data `(1, 1800, 3600)`, time `(100,)` = one commit batch, `lat`/`lon` single |
| Chunk inventory | 235,203 of 235,228 written, none misplaced, none degenerate or truncated; the 25 absent are legitimate (finding 7) |
| Values vs source netCDF | 144 planes bit-identical, NaN pattern ≡ raw fill pattern, each on the day raw says |
| File seams | all 45 year boundaries clean on both sides, value and date |
| Component identity | `E = Eb+Ec+Ei+Es+Et+Ew` closes to 4.8e-07 |
| Physical bounds | all 14 within the bounds their units imply |
| Snapshots | 170 reachable = 169 batches + the initial one |
| Compressed size | 1.032 TiB, 0.186x logical |

Two of these are worth more than their cost. The **chunk sweep** is the only
check here with full coverage: `session.chunk_coordinates` and
`session.store.getsize` both run off the manifest, so all 235,228 chunks can be
audited for existence and size in 32 seconds without reading a byte of data. A
plane that came out constant or truncated compresses orders of magnitude
smaller than a real field, so size alone localises it. Any future rebuild
should run this phase before anything else — it is close to free and it is the
only thing that looks everywhere.

The **component identity** is the cheapest real check on the merge. `E` is the
sum of its six components, and that sum closing to float32 rounding across a
whole plane is only possible if all seven variables were put on the same grid
cell at the same timestep. Four days of it is worth more than any number of
additional random planes, which can only ever re-confirm one variable at a
time.

### 7. `E` is entirely absent on 25 days, and that is correct

The sweep found `E` holding 16,777 chunks against 16,802 timesteps. Zarr does
not write a chunk whose cells are all equal to the fill value, so a hole reads
back as all-NaN — and raw `E` is entirely -999 on exactly those 25 days:

| Block | Store timesteps |
| --- | --- |
| 1982-02-05 .. 02-09 | 766-770 |
| 1992-01-01 .. 01-05 | 4383-4387 |
| 1992-03-21 .. 03-25 | 4463-4467 |
| 1992-05-20 .. 05-24 | 4523-4527 |
| 1992-10-02 .. 10-06 | 4658-4662 |

Five blocks of five consecutive days. Every other variable — including E's own
components `Et`, `Eb`, `Ei` — has full data on those dates, so `E` cannot equal
its component sum there. This is an **upstream GLEAM v4.3a gap**, and `E` is
the only one of the 14 variables with any fully-absent day.

The lesson for the audit is that a chunk count below `n_time` is not by itself
a defect. `check_sweep` now traces every hole back to raw and fails only if raw
holds real data there — which is the difference between reporting a finding and
triggering a pointless 57 core-hour rebuild.

### 8. Two invariants that the data does not actually have

Both of these were asserted, both failed, and neither was the store's fault.
Both were settled the same way: recompute the property from raw and see whether
the store reproduces it exactly.

**`Ep = Ep_aero + Ep_rad` does not close.** It misses by up to 0.48 on about 20
of 2.2 million valid cells per day (0.001%), while the mean residual stays at
1e-6. The residual field computed from the store is **bit-identical** to the
residual field computed from raw, and at the worst cell all three values match
raw exactly: `Ep=4.0` against `aero+rad=4.483`. GLEAM appears to cap `Ep`
around 4 mm/day in some cells. Upstream, faithfully reproduced.

**The valid-cell footprint is not a static land mask.** It moves day to day for
11 of the 14 variables — `SMrz` drifts across 2,198,635 .. 2,203,562 valid
cells over six sampled days. Only `Eb`, `Ec` and `Et` are static, at 2,210,131.
On every day checked the store's NaN pattern is identical to raw's fill
pattern, counts included. Upstream again.

The script now asserts the bounds but only *reports* the footprint, confirming
against raw on the day it is widest apart. Asserting a static mask would have
reported normal GLEAM behaviour as corruption on 11 of 14 variables every time
it ran.

A third assertion was simply mis-designed: the bounds and the mask were bundled
into one check, so 11 range checks that all passed were reported as failures by
association. Worth remembering when reading any of these tables — one
assertion, one property.

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
- **The 8.57 GiB of orphaned chunks was left in place.** `expire_snapshots` +
  `garbage_collect` would reclaim it, but it is 0.8% of a 1.041 TiB store
  against ~108 TiB of free scratch, and the dry run reports it would also
  delete all 171 snapshots and transaction logs — i.e. the entire commit
  history. That may be by design, since the intermediate per-batch snapshots
  have little value once the store is verified, but it is not an irreversible
  operation worth running on the only copy of a freshly built deliverable for a
  0.8% saving. Revisit before archiving, or if repeated rebuilds accumulate
  orphans (~4-5 GB per killed batch).
- **The store lives on scratch, which is purged.** 1.041 TiB sitting in
  `/glade/derecho/scratch` is subject to the purge policy; the campaign
  allocation `/glade/campaign/univ/umic0112` has 5 TiB free and 0 used. Moving
  or copying the finished store there matters considerably more than reclaiming
  8.57 GiB.
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

The verification of the finished store *is* committed, as
`verify_gleam_zarr.py`. It reads the same config as the build, opens the store
read only, and never calls anything destructive except
`garbage_collect(dry_run=True)`. Run against the tiny config it is a few
seconds, which makes it a reasonable smoke test after any change to the
encoding or chunking logic:

```bash
# everything: ~10 minutes on a login node, one core, free
uv run python verify_gleam_zarr.py --config config/config_zarr.yaml

# the metadata-only phases: seconds, and the sweep is the only full-coverage
# check there is -- run this first after any rebuild
uv run python verify_gleam_zarr.py --config config/config_zarr.yaml \
    --phases structure,index,sweep

# spend longer on value comparisons
uv run python verify_gleam_zarr.py --config config/config_zarr.yaml \
    --phases samples --samples 140 --seed 1
```

It exits non-zero if any check fails, and writes to
`logs/verify_gleam_zarr.log` as well as stdout. `--seed` is worth varying
between runs: the stratified draw is reproducible by default, which is
convenient for comparing runs and useless for finding something new.

Per-batch timings can be read straight out of the log — the pipeline already
logs a timestamped line when a batch starts writing and another when it
commits. Actual core utilisation comes from `qhist -u $USER -c` (the `Avg CPU`
column is percent of allocated cores).
