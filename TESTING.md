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
and was **not** cleaned up at the time.

The 171 orphaned *snapshots* are a different story, and reading them as the
commit history is what deferred the cleanup for a day. They were collected on
2026-09-10 — finding 9.

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

## Finalization (2026-09-10)

Turning the verified store into a deliverable: provenance attributes, an
immutable tag, and the orphan cleanup this document had deferred. All of it ran
on a login node for **free**, in about six minutes of wall time.

| Step | Result |
| --- | --- |
| Global attributes | 7 -> 29, commit `N9M19WV4ZTGTPXBJA690` |
| Tag | `v4.3a-verified-20260910` at that commit |
| Garbage collection | 8.57 GiB, 2012 chunks, 513 manifests, 171 snapshots, 171 transaction logs |
| Store on disk | 1,144,508,450,624 -> 1,135,311,017,909 bytes (**8.567 GiB freed**) |
| Reachable | 1.0321 TiB across 171 snapshots, **unchanged either side of collection** |
| Re-verification | 65 checks, 0 failures on `structure,sweep`; unreachable now 0.00 GiB |

The store had only ever carried GLEAM's own seven attributes, so nothing in it
recorded what produced it, from what, when, that it had been verified, or that
`E` is absent on 25 days. It now carries 29, including a `chunking` note saying
what the `time: 1` layout is and is not good for, and `known_data_gaps`. The
coverage and grid attributes are derived from the data by `derive_attrs` rather
than written by hand, so they cannot drift from what was actually stored; the
rest come from a new `attrs` section in the config, so they are in git and
reusable by the next rebuild.

`Conventions` is **`ACDD-1.3`, not CF**. Claiming CF would have been false: the
variables' `standard_name` values are descriptive upstream strings (`'Actual
evaporation'`, not `water_evaporation_flux`) and the units are spelled
`mm.day-1` and `m3.m-3`, which udunits does not parse. A `cf_compliance`
attribute says so in the store rather than leaving a consumer to find out.

### 9. The 171 orphaned snapshots are `fork` snapshots, one per commit

The claim this document made — that collection "would also delete all 171
snapshots and transaction logs — i.e. the entire commit history" — was wrong,
and the counts on disk were enough to disprove it without running anything:

| directory | before | after |
| --- | --- | --- |
| `chunks` | 237,553 | 235,541 |
| `manifests` | 3,386 | 2,873 |
| `snapshots` | 342 | **171** |
| `transactions` | 342 | **171** |
| `overwritten` | 342 | 343 |

`snapshots/` held 342 objects while `ancestry(branch='main')` yielded 171, so
342 = 171 reachable + 171 orphaned, and the 171 in the summary were the orphans.
Collection took the directory down to exactly the reachable count. Nothing in
live history was ever in the delete set: `garbage_collect` deletes an object
only if no surviving snapshot references it, and `expire_snapshots` — the call
that *would* collapse history — was never involved and is not needed to reclaim
anything.

What the orphans actually are is the more useful half. Looking one up gives the
answer immediately: every one of the 171 has the commit message **`'fork'`**.
`to_icechunk` forks the writable session for a dask-backed write, and the fork's
snapshot is not part of the branch's ancestry, so **every batch leaves one
unreachable snapshot behind as a matter of course** — 169 committed batches plus
the two in flight when jobs 1 and 2 were killed at walltime is exactly 171. The
same two killed batches account for the 2012 orphaned chunks. So the snapshot
orphans are structural and the chunk orphans are the kill damage, which is why
one number scaled with the run's length and the other did not.

That was confirmed before touching the real store, on a synthetic repository
small enough to run in seconds: three dask-backed batches produced 7 snapshot
objects — 4 reachable, 3 forks — and a session abandoned without committing
added one more snapshot and one orphaned chunk. Collection removed exactly those
4 snapshots and 1 chunk, left `snapshots/` at the 4 reachable, and the data came
back bit-identical. A first attempt at the same rehearsal wrote **numpy**-backed
data and produced no forks at all, which is itself the proof that the fork is a
property of the dask write path rather than of committing.

**The lesson is about the measurement, not the store.** A `GCSummary` count read
on its own is unfalsifiable — "171 snapshots" is equally consistent with
"the whole history" and "171 objects nothing points at". It only becomes
meaningful next to `len(ancestry(...))` and the object count on disk, and those
two commands would have cost seconds. The missing comparison deferred a safe,
free cleanup and put a wrong conclusion in this file.

That is now a single command rather than a habit to remember:
`finalize_gleam_zarr.py --status` prints the three numbers together — reachable
history, objects on disk, and what is unreachable — and is step 1 of the
documented procedure for exactly this reason.

Two smaller things worth keeping:

- **`overwritten/` is not self-cleaning.** It holds superseded copies of the
  `repo` ref file, one per mutating operation, and collection *added* one rather
  than removing any — it went 342 -> 343. It is 4.9 MB and nothing references
  it, so it is cosmetic, but it does grow without bound across rebuilds.
- **Collection is idempotent.** A second `--gc --apply` reports zeros, which is
  the cheapest confirmation that the first one converged.

### 10. Finalization belongs in its own script, not behind a verifier flag

`--gc` was deliberately *not* added to `verify_gleam_zarr.py`. The verifier is
what proves a finalization step did no harm, so a tool that both acted and
audited would answer two questions with one exit status — and the read-only
promise in its module docstring is the thing that makes it safe to run by
reflex. `finalize_gleam_zarr.py` takes exactly one action per invocation and
writes nothing without `--apply`.

Its `--gc` brackets the collection with the two properties collection must not
change, reachable bytes and history length, and exits non-zero if either moves.
That check is the whole reason it was comfortable to run an irreversible
operation on a store with no second copy — that, and the rehearsal above.

## Building the temporal store (2026-09-11)

Record of the testing for the **second** store, chunked
`(time, lat, lon) = (16802, 20, 20)` for time-series reads, built from the same
raw files by the new `region` write strategy. Same tiered method as 2026-09-09
and for the same reason; what it found was again not what it went looking for.

The two findings below are both cases where the volume of data moved was
identical and the cost was not. Neither is visible in a profile of the pipeline
— one is a property of how the source files are laid out, the other of how dask
assembles an array — and neither shows up on a small or sparse test fixture.

### 11. A block narrower than the globe reads the file strided, not sequentially

The obvious way to bound the memory a region block takes is to shrink it in both
lat and lon. That is wrong, and it is wrong for a reason that volume arithmetic
cannot see.

The raw files are chunked `[12, 57, 113]`. A block spanning all of lon touches
every lon chunk, so each `(time chunk, lat chunk)` pair is read as one
contiguous run of 32 source chunks. A block half as wide touches 11 of every 32,
**scattered through the file**, and the same bytes then arrive as several times
as many seeks.

Measured on one `E` year file, cold, 512 MiB chunk cache:

| read | wall | volume | rate |
| --- | --- | --- | --- |
| `[:, 0:600, 0:1200]` strided in lon | 29.4 s | 0.98 GiB | **34.2 MiB/s** |
| `[:, 0:600, :]` full lon | 29.7 s | 2.95 GiB | **101.7 MiB/s** |
| `[0:67, :, :]` whole planes, for reference | 10.9 s | 1.62 GiB | 151.4 MiB/s |

3.0x for the same bytes, purely from locality. (A fourth row, `[:, 0:200, :]` at
341.7 MiB/s, is not evidence — that band was already in page cache from the two
reads above it. It is left out of the conclusion.)

**How it was nearly missed.** The shape was first tried at `600 x 1200` on `Ec`,
which compresses about 70:1 and finished in seconds per block. On `E`, which
compresses about 3.4:1, the same shape sat at **9% cpu with no block committed
in eleven minutes**. The fixture was not small, it was *sparse*, and a sparse
variable barely touches the disk. **Test a read-pattern change on the dense
variable.** And when a rechunk looks slow, read cpu% first: 9% says locality,
not codec.

`block_shape.lon` is now required to be `-1`, and the cost of the block is paid
in lat alone: `lat: 200` is 45.1 GiB resident against 1.43x read amplification,
which is the right trade when Derecho bills cpus and not memory.

### 12. `.load()` doubles a block at the moment it comes together

With the block shape fixed, the tier-2 bench still died — on its **first block**,
after 59 minutes, with `RuntimeError: NetCDF: HDF error`.

That error is a red herring twice over. It is not an HDF5 defect, and it is not
really about netCDF at all. The tell is in the timings: **46 minutes of system
time against 2 minutes of user**. A process that spends 95% of itself in the
kernel is not computing, it is being ground through page reclaim against a
memory cgroup, and the allocation that finally fails is simply whichever one
came next — here, one inside HDF5, which reports it uninformatively.

The budget against the job's 96 GB:

| | |
| --- | --- |
| HDF5 chunk caches, 46 open files x 512 MiB | 23.6 GB |
| block buffer | 48.4 GB |
| dask holding all 46 inputs *and* allocating the concatenated output | +48.4 GB |
| **total** | **~120 GB** |

`.load()` is the obvious way to materialise a block and the wrong one at this
size: dask holds every input chunk while it allocates the output beside them, so
a 45 GiB block is transiently 90 GiB with nothing in the log to say so.
`read_into_buffer` fills one preallocated array a time chunk at a time instead.
The chunk boundaries are the file boundaries, so every read is still a single
contiguous hyperslab out of one file, and peak memory becomes the block plus one
slab.

`chunk_cache_size_mib` was the second term, and it was inherited rather than
chosen: 512 is the 7.6x lever on the `append` path, where the store is written
one timestep at a time out of twelve-deep source chunks. On the `region` path it
is not doing that job — a block is one hyperslab and every source chunk it
touches is decompressed once regardless — but it still multiplies by the open
file count, and a region job holds all 46 files of a variable open. At 64 MiB it
costs 2.9 GB.

Projected peak falls from ~120 GB to **49 GiB**. The append-path configs keep
512, where it is still worth 7.6x.

**The generalisation worth keeping:** on the append path peak memory is the
chunks in flight, and `configure_runtime` bounds it. On the region path peak
memory *is the block*, and the two settings that used to be the memory story are
now minor next to it. The strategies do not share a cost model, and a setting
carried from one to the other should be re-derived, not inherited.

### Tier 2 outcome: Ep, all 46 years (job 7403373, 2026-09-11)

Nine blocks, 0.40 TiB logical, **137 min**, exit 0. Tier 1 rebuilt and verified
clean in the same job (27 checks, 0 failures).

| | |
| --- | --- |
| per block (45.1 GiB logical) | 11.4 - 19.6 min, mean 15.2 |
| throughput | 0.0455 GiB/s logical |
| spatial build, for comparison | 0.111 GiB/s |
| cpu actually used | **0.47 cores**, while billing 2 |
| `Ep` compressed | **123 GiB**, against 123.8 GiB for the same variable in the spatial store |

**The region path is 2.4x slower per logical byte than the append path**, and
that is explained rather than mysterious: 1.43x read amplification from the lat
band, times 1.49x worse locality than a whole-plane read (101.7 against
151.4 MiB/s in the finding 11 probe). 1.43 x 1.49 = 2.13, near enough.

**The layout buys no space.** 123 GiB against 123.8 GiB for the same variable.
The ocean tiles that vanish entirely here were already compressing to almost
nothing in the spatial store, so the saving that looked available from the chunk
counts -- 43% of the grid written on the tiny fixture -- is not a saving in
bytes. Budget the finished temporal store at ~1.03 TiB, the same as its sibling.

### Sizing the production chain from two runs

Cost was fitted as `a x (raw bytes decompressed) + b x (logical bytes written)`
against the two full-scale runs there are -- the spatial production build
(1097 GB raw, 5686 GB logical, 14.2 h) and this bench (127 GB raw x 1.43, 406 GB
logical, 2.29 h):

    a = 12.3 h per TB decompressed      b = 0.12 h per TB written

Decompressing the raw files is ~97% of it, which is why scaling by *logical*
volume overestimates so badly: the 14 variables are identical in logical size
but range from 6.2 GB to 128 GB of raw bytes, and the sparse ones are nearly
free. Pure logical scaling says 32 h; the fit says **20 h**.

| lat band | read amplification | projected wall | core-hours at ncpus=1 |
| --- | --- | --- | --- |
| 200 (benched) | 1.43x | 20.0 h | 20.0 |
| 360 | 1.27x | 17.9 h | 17.9 |
| 600 | 1.14x | 16.1 h | 16.1 |

### 13. `Used Mem` in qhist is not peak RSS

The bench reported `Used Mem = 96.00000381469727` against a `mem=96GB` request,
which reads like the job pressing against its ceiling. It is not. The cgroup
counter includes reclaimable page cache, which a job reading a terabyte of
netCDF fills to the limit as a matter of course. The same column reads exactly
`32.0` for the 2026-09-09 production jobs, whose real high-water was 16.8 GB.

The number is only diagnostic when a job **fails**: job 7401899 died reporting
66.13 GB against the same 96 GB cap, and that is the tell -- not that it reached
the cap, but that it did not, because the allocation that was refused (a ~48 GB
concatenation buffer on top of 66 GB already resident) never became resident to
be counted. Read it with the sys/user ratio beside it: 78% of wall in system
time for the failed run against 18% for the healthy one.

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
- ~~**The 8.57 GiB of orphaned chunks was left in place.**~~ **Done
  2026-09-10**, and the reasoning that deferred it was wrong. See finding 9.
- **The store lives on scratch, which is purged.** 1.041 TiB sitting in
  `/glade/derecho/scratch` is subject to the purge policy; the campaign
  allocation `/glade/campaign/univ/umic0112` has 5 TiB free and 0 used.
  **Accepted, 2026-09-10**: the store stays on scratch for now and the move is
  handled separately. `directories.raw` was added so a moved store can still be
  verified against a raw tree that did not move with it, which is the coupling
  the move runs into first. Until it moves, treat the store as reproducible
  rather than archived.
- ~~**`chunks: {time: 1}` is the worst possible layout for point time-series
  reads**, which touch all 16802 chunks of a variable.~~ **Being addressed
  2026-09-11**: a second store chunked `(16802, 20, 20)` is built beside it
  rather than changing this one, which stays correct for maps. See *Building
  the temporal store*. Both layouts are recorded in their stores' own
  `chunking` attributes, each pointing at the other, so a consumer does not
  have to discover either.
- **The store is not CF compliant, and is not claimed to be.** Each variable's
  `standard_name` and `units` come through unaltered from upstream, so the
  standard names are descriptive strings and the units (`mm.day-1`, `m3.m-3`)
  are not udunits-parseable. `Conventions` is `ACDD-1.3` and `cf_compliance`
  states the gap. Fixing it means rewriting per-variable metadata and diverging
  from what GLEAM published, which is a decision about what the store *is*
  rather than a defect — worth taking deliberately if anything downstream runs
  a CF checker.
- **`overwritten/` grows without bound.** One superseded `repo` ref file per
  mutating operation, 4.9 MB today, unreferenced and not collected. Cosmetic,
  but it accumulates across rebuilds.

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
`logs/verify_gleam_zarr.log` as well as stdout.

Finalization is separate, and writes nothing without `--apply`:

```bash
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --attrs
uv run python finalize_gleam_zarr.py --config config/config_zarr.yaml --gc
```

Run either without `--apply` first — the output is exactly what the applied run
will do. After a `--gc --apply`, re-run the verifier's `structure,sweep` phases:
they are the full-coverage check that nothing reachable was touched. `--seed` is worth varying
between runs: the stratified draw is reproducible by default, which is
convenient for comparing runs and useless for finding something new.

Per-batch timings can be read straight out of the log — the pipeline already
logs a timestamped line when a batch starts writing and another when it
commits. Actual core utilisation comes from `qhist -u $USER -c` (the `Avg CPU`
column is percent of allocated cores).
