#!/usr/bin/env bash
#SBATCH --job-name=talos-e2e
#SBATCH --partition=gpuquick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
# End-to-end `talos run` over AIR-MOT val sequences, timing every stage.
# Submit from the repo root:
#   cd ~/talos/repo && SEQS="AIRMOT_plane/airmot-043 AIRMOT_ship/airmot-099" TAG=a100 \
#     sbatch --export=ALL jobs/run_e2e.sbatch
# Env: SEQS (bench/seq ...), TAG (label for this hardware), DEVICE (cuda:0 | cpu),
#      ZONE ("cx,cy,r", default frame centre r=150), BATCH (4).
source "$HOME/talos/repo/jobs/common.sh"
: "${SEQS:?}" "${TAG:?}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true
lscpu | grep -E "Model name|^CPU\(s\)" || true
for s in $SEQS; do
  name=$(basename "$s")
  python -m src.TALOS.pipeline.run "$DATA/mot/$s/img1" --out "$TALOS/runs/e2e/$TAG/$name" \
    --device "${DEVICE:-cuda:0}" --zone-circle "${ZONE:-960,540,150}" --batch "${BATCH:-4}"
  mkdir -p "results/e2e/$TAG"
  cp "$TALOS/runs/e2e/$TAG/$name/timing.json" "results/e2e/$TAG/$name.json"
done
echo "E2E DONE"
