#!/usr/bin/env bash
#SBATCH --job-name=talos-space-test
#SBATCH --partition=cpucluster
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm-%j.out
# Build the Space and smoke-test it on 2 CPUs (the free Hugging Face tier's size), in the
# Space's own pinned environment (~/talos/.venv-space, built from space/requirements.txt).
#   cd ~/talos/repo && sbatch jobs/space_test.sbatch
set -euo pipefail
cd "$HOME/talos/repo"
bash scripts/build_space.sh "$HOME/talos/build/space"
cd "$HOME/talos/build/space"
export OMP_NUM_THREADS=2
"$HOME/talos/.venv-space/bin/python" "$HOME/talos/repo/scripts/test_space.py" \
  --clip-frames "$HOME/talos/data/mot/VISO_plane-all/viso-043/img1" --zone 60,40,40
echo "SPACE TEST DONE"
