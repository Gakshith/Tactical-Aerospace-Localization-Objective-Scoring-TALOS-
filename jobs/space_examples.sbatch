#!/usr/bin/env bash
#SBATCH --job-name=talos-space-ex
#SBATCH --partition=gpuquick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
# Pre-render the Hugging Face Space's VISO example clips (see scripts/space_examples.py).
#   cd ~/talos/repo && sbatch jobs/space_examples.sbatch
source "$HOME/talos/repo/jobs/common.sh"
PYTHONPATH="$REPO" python scripts/space_examples.py --out "$TALOS/space_examples"
du -sh "$TALOS/space_examples"
echo "SPACE EXAMPLES DONE"
