#!/bin/bash -l
#
# Submit the GLEAM zarr build as a Casper batch job.
#
#   ./submit_gleam_zarr.sh                          # default config
#   CONFIG=config/other.yaml ./submit_gleam_zarr.sh
#
# The account is taken from $PBS_ACCOUNT, the same variable qcmd and
# qinteractive read, so it can live in ~/.bashrc rather than in this file.
# Passing it explicitly still works: qsub -A <PROJECT> submit_gleam_zarr.sh
#
# The build is restartable: a run killed by walltime leaves a store committed up
# to its last batch, and resubmitting this script with the same config picks up
# from there. Resubmitting after a completed run is a no-op.
#
#PBS -N gleam_zarr
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=32GB
#PBS -l walltime=12:00:00
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
    qsub_args=(-A "${PBS_ACCOUNT}")
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
echo "cpus     ${NCPUS:-unset}, memory ${PBS_MEM:-see select}"

# num_workers lives in the config while ncpus lives here, so they can drift; an
# oversubscribed run would multiply its chunks in flight straight past the
# memory the job asked for
workers=$(uv run python -c "
from utils.path_utils import load_config
print(load_config('${CONFIG}').get('num_workers') or '')
")
if [[ -n ${workers} && -n ${NCPUS:-} ]] && (( workers > NCPUS )); then
    echo "error: num_workers=${workers} in ${CONFIG} exceeds ncpus=${NCPUS}" >&2
    echo "       raise ncpus in this script or lower num_workers in the config" >&2
    exit 1
fi

uv run python gleam_zarr.py --config "${CONFIG}"

echo "finished $(date)"
