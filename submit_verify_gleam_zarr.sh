#!/bin/bash -l

#PBS -N gleam_verify
#PBS -j oe
#PBS -o logs/

set -euo pipefail

# Verify a whole layout's 14 stores against the raw netCDF files, in parallel.
#
# One store at a time on a login node is what the README documents, and that is
# right for one store: ~9 minutes spatial, ~15 temporal, free. A whole family is
# 28 of those, ~5.6 hours of reading across 2.1 TB, which is not login node
# work. This runs the 14 stores of one layout as 14 processes.
#
# It is read only. Unlike a build, nothing here holds the icechunk branch, so
# the two layouts' jobs can run at the same time and need no dependency between
# them.
#
#   CONFIG=config/config_zarr_spatial.yaml \
#   COMPARE_WITH=config/config_zarr_temporal.yaml ./submit_verify_gleam_zarr.sh

# stock qsub reads only PBS_DEFAULT and PBS_DPREFIX from the environment, and a
# #PBS line is a literal comment that cannot expand $PBS_ACCOUNT, so the account
# is resolved here instead: run outside PBS, this hands itself to qsub with it
if [[ -z ${PBS_ENVIRONMENT:-} ]]; then
    if [[ -z ${PBS_ACCOUNT:-} ]]; then
        echo "error: PBS_ACCOUNT is not set" >&2
        echo "       export it in ~/.bashrc, or submit with qsub -A <PROJECT>" >&2
        exit 1
    fi
    if [[ -z ${CONFIG:-} ]]; then
        echo "error: set CONFIG to the layout's config" >&2
        exit 1
    fi
    # develop rather than main: main bills the whole node's 128 cpus whatever
    # the job uses, and this wants 14. develop is shared and bills what it asks
    # for, at the cost of a 6 hour walltime cap -- ample here.
    QUEUE="${QUEUE:-develop}"
    NCPUS="${NCPUS:-14}"
    # the same 28 stores verified in ~25 min per layout on this shape
    # (TESTING.md, "Verification and finalization outcome"). 2 h is 4.8x that,
    # and since develop bills ncpus x wall time, the request is also the ceiling
    # on what a hung job can cost: 14 x 2 h = 28 core-hours rather than 84.
    WALLTIME="${WALLTIME:-2:00:00}"
    # Derecho bills cpus only, so memory is free and should be asked for freely
    # (TESTING.md finding 4: a 10 GB default cost 5.2x in re-decompression).
    # The verifier does not call configure_runtime, so it takes xarray's default
    # file_cache_maxsize of 128 and HDF5's 64 MiB chunk cache per open file --
    # up to ~8 GB per process before any data. Fourteen of those is the number
    # that matters here, not the ~26 MiB box each one is actually comparing.
    MEM="${MEM:-120GB}"

    qsub_args=(
        -A "${PBS_ACCOUNT}"
        -q "${QUEUE}"
        -l "select=1:ncpus=${NCPUS}:mem=${MEM}"
        -l "walltime=${WALLTIME}"
    )
    # job_priority is a main queue concept; the develop queue rejects it
    if [[ ${QUEUE} == main ]]; then
        qsub_args+=(-l job_priority=regular)
    fi
    qsub_args+=(-v "CONFIG=${CONFIG},COMPARE_WITH=${COMPARE_WITH:-},PHASES=${PHASES:-}")
    # an absolute path so the submission does not depend on the caller's cwd
    exec qsub "${qsub_args[@]}" "$(readlink -f "$0")"
fi

# qsub starts the job in $HOME; PBS_O_WORKDIR is where it was submitted from
cd "${PBS_O_WORKDIR:-$(dirname "$(readlink -f "$0")")}"

# the same module set the venv was built against, so the wheels' bundled HDF5
# does not meet a different one through LD_LIBRARY_PATH
module reset > /dev/null 2>&1

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="${TMPDIR:-/glade/derecho/scratch/$USER/tmp}"
mkdir -p "$TMPDIR" logs

# each process is its own serial reader; a threaded BLAS would oversubscribe
export OMP_NUM_THREADS=1
export MALLOC_TRIM_THRESHOLD_=0

# all seven phases. metadata is not in the default set because it needs a
# sibling config, and it is exactly what a nomenclature change has to be held
# to: it requires the two layouts of a variable to agree, attribute for
# attribute, on what they hold and what it is called.
PHASES="${PHASES:-structure,index,samples,sweep,identities,ranges,metadata}"

echo "job      ${PBS_JOBID:-interactive} on $(hostname)"
echo "started  $(date)"
echo "config   ${CONFIG}"
echo "compare  ${COMPARE_WITH:-none}"
echo "phases   ${PHASES}"
echo "queue    ${PBS_QUEUE:-unset}"

# the cpu count comes from the affinity mask rather than $NCPUS or nproc, both
# of which lie here: on a shared develop node PBS reports NCPUS=1 whatever it
# granted, and nproc reports OMP_NUM_THREADS, which this script pins to 1
read -r available family < <(uv run python -c "
import os
from utils.path_utils import load_config, resolve_variables
print(len(os.sched_getaffinity(0)),
      ' '.join(resolve_variables(load_config('${CONFIG}'))))
")
VARIABLES="${VARIABLE:-${family}}"
n_processes=$(wc -w <<< "${VARIABLES}")

echo "verifying ${n_processes} stores on ${available} cpus"
if (( n_processes > available )); then
    echo "error: ${n_processes} stores on ${available} cpus would oversubscribe;" >&2
    echo "       raise ncpus or pass VARIABLE to verify fewer" >&2
    exit 1
fi

compare_args=()
if [[ -n ${COMPARE_WITH:-} ]]; then
    compare_args=(--compare-with "${COMPARE_WITH}")
fi

pids=""
for variable in ${VARIABLES}; do
    # one log per store, or fourteen verifiers interleave into one unreadable
    # file and a failure cannot be attributed
    uv run python verify_gleam_zarr.py \
        --config "${CONFIG}" \
        --variable "${variable}" \
        --phases "${PHASES}" \
        "${compare_args[@]}" \
        > "logs/verify_$(basename "${CONFIG}" .yaml)_${variable}.log" 2>&1 &
    pids="${pids} $!:${variable}"
done

status=0
for entry in ${pids}; do
    if wait "${entry%%:*}"; then
        echo "ok     ${entry##*:}"
    else
        echo "FAILED ${entry##*:}   see logs/verify_$(basename "${CONFIG}" .yaml)_${entry##*:}.log"
        status=1
    fi
done

echo "finished $(date)"
# non-zero if any store failed, so the chain and the caller both see it
exit ${status}
