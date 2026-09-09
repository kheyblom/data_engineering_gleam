#!/bin/bash -l

#PBS -N gleam_zarr
#PBS -j oe
#PBS -o logs/

set -euo pipefail

# stock qsub reads only PBS_DEFAULT and PBS_DPREFIX from the environment, and a
# #PBS line is a literal comment that cannot expand $PBS_ACCOUNT, so the account
# is resolved here instead: run outside PBS, this hands itself to qsub with it
if [[ -z ${PBS_ENVIRONMENT:-} ]]; then
    if [[ -z ${PBS_ACCOUNT:-} ]]; then
        echo "error: PBS_ACCOUNT is not set" >&2
        echo "       export it in ~/.bashrc, or submit with qsub -A <PROJECT>" >&2
        exit 1
    fi
    # the queue shape is not a #PBS directive either, since testing needs to
    # vary it and a directive cannot expand a variable: these defaults are the
    # production run, and a develop queue test overrides them in the
    # environment. They are set only here, so that inside the job NCPUS stays
    # whatever PBS actually granted and the guard below can trust it.
    QUEUE="${QUEUE:-main}"
    NCPUS="${NCPUS:-128}"
    # derecho caps the main queue at 12 hours; a run too big for that resumes
    WALLTIME="${WALLTIME:-12:00:00}"
    # a main queue job gets the whole node's memory and needs no request; a
    # shared develop job gets a flat 10 GB default whatever ncpus it asked for,
    # which is less than the chunk caches want and quietly starves the run into
    # re-decompressing everything, so a test has to ask for memory explicitly
    MEM="${MEM:-}"

    select="1:ncpus=${NCPUS}"
    if [[ -n ${MEM} ]]; then
        select="${select}:mem=${MEM}"
    fi

    qsub_args=(
        -A "${PBS_ACCOUNT}"
        -q "${QUEUE}"
        -l "select=${select}"
        -l "walltime=${WALLTIME}"
    )
    # job_priority is a main queue concept; the develop queue rejects it
    if [[ ${QUEUE} == main ]]; then
        qsub_args+=(-l job_priority=regular)
    fi
    # only one writer at a time can hold the icechunk branch, so a run too long
    # for one walltime chains its jobs instead of overlapping them. The default
    # is afterany, not afterok: a job stopped by walltime exits non-zero, and
    # that is precisely the case the next job in the chain exists to resume.
    DEPEND="${DEPEND:-afterany}"
    if [[ -n ${AFTER:-} ]]; then
        qsub_args+=(-W "depend=${DEPEND}:${AFTER}")
    fi
    if [[ -n ${CONFIG:-} ]]; then
        qsub_args+=(-v "CONFIG=${CONFIG}")
    fi
    # an absolute path so the submission does not depend on the caller's cwd
    exec qsub "${qsub_args[@]}" "$(readlink -f "$0")"
fi

# qsub starts the job in $HOME; PBS_O_WORKDIR is where it was submitted from
cd "${PBS_O_WORKDIR:-$(dirname "$(readlink -f "$0")")}"

# the same module set the venv was built against, so the wheels' bundled HDF5
# does not meet a different one through LD_LIBRARY_PATH
module reset > /dev/null 2>&1

export PATH="$HOME/.local/bin:$PATH"
# keep scratch space, not /tmp, behind anything the libraries spill
export TMPDIR="${TMPDIR:-/glade/derecho/scratch/$USER/tmp}"
mkdir -p "$TMPDIR" logs

# dask supplies the parallelism; a threaded BLAS underneath it would oversubscribe
export OMP_NUM_THREADS=1
# return freed chunk buffers to the OS rather than holding them in the heap,
# which matters when the job is sized close to its high water mark
export MALLOC_TRIM_THRESHOLD_=0

CONFIG="${CONFIG:-config/config_zarr.yaml}"

echo "job      ${PBS_JOBID:-interactive} on $(hostname)"
echo "started  $(date)"
echo "config   ${CONFIG}"
echo "queue    ${PBS_QUEUE:-unset}"
# the true cpu count is reported by the guard below, which already starts python

# num_workers lives in the config while the job size lives here, so they can
# drift; an oversubscribed run would multiply its chunks in flight straight past
# the memory the job asked for.
#
# The cpu count comes from the affinity mask rather than $NCPUS or nproc, both of
# which lie here: on a shared develop node PBS reports NCPUS=1 whatever it
# granted, and nproc reports OMP_NUM_THREADS, which this script pins to 1.
read -r workers available < <(uv run python -c "
import os
from utils.path_utils import load_config
print(load_config('${CONFIG}').get('num_workers') or 0, len(os.sched_getaffinity(0)))
")
echo "workers  num_workers=${workers:-unset}, cpus available=${available}"
if (( workers > available )); then
    echo "error: num_workers=${workers} in ${CONFIG} exceeds the ${available} cpus" >&2
    echo "       this job can use; raise ncpus or lower num_workers" >&2
    exit 1
fi

uv run python gleam_zarr.py --config "${CONFIG}"

echo "finished $(date)"
