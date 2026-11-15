# Sourced by every TALOS v2 Slurm job:  source "$HOME/talos/repo/jobs/common.sh"
set -euo pipefail

export TALOS="$HOME/talos"
export REPO="$TALOS/repo"
export DATA="$TALOS/data"
export WEIGHTS="$TALOS/weights"
export LOGS="$TALOS/logs"
export PATH="$HOME/.local/bin:$PATH"

# venv activate scripts reference unset vars; relax -u just for that.
set +u
source "$TALOS/.venv/bin/activate"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export YOLO_OFFLINE=1   # compute nodes: never try to auto-download

mkdir -p "$LOGS"
cd "$REPO"

echo "[common] host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-unset} python=$(command -v python)"
