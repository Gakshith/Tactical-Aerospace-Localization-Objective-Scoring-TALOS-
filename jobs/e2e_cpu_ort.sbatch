#!/usr/bin/env bash
#SBATCH --job-name=talos-e2e-cpu
#SBATCH --partition=cpucluster
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
# Day 5: the whole pipeline on CPU with ONNX Runtime, on the same 40 frames of airmot-043
# as the PyTorch CPU baseline (results/e2e/cpu32). Run after jobs/export_parity.sbatch:
#   cd ~/talos/repo && sbatch --dependency=afterok:<export job id> jobs/e2e_cpu_ort.sbatch
# Rows: fp32 and dynamic-int8 graphs at 32 intra-op threads, and fp32 at 2 threads
# (the size of a free Hugging Face CPU Space; same node, so only the ORT threads are capped).
source "$HOME/talos/repo/jobs/common.sh"
EXP="$WEIGHTS/export"
SEQ="$DATA/mot/AIRMOT_plane-all/airmot-043/img1"
lscpu | grep -E "Model name|^CPU\(s\)" || true
run() {  # tag model threads
  python -m src.TALOS.pipeline.run "$SEQ" --out "$TALOS/runs/e2e/$1/airmot-043" --device cpu \
    --zone-circle 960,540,150 --batch 4 --max-frames 40 --no-video \
    --backend onnx --ort-provider cpu --weights "$2" --threads "$3"
  mkdir -p "results/e2e/$1"
  cp "$TALOS/runs/e2e/$1/airmot-043/timing.json" "results/e2e/$1/airmot-043.json"
}
run cpu32_ort_fp32 "$EXP/y26s_t1024_u8.onnx" 32
run cpu32_ort_int8dyn "$EXP/y26s_t1024_int8dyn.onnx" 32
OMP_NUM_THREADS=2 run cpu2_ort_fp32 "$EXP/y26s_t1024_u8.onnx" 2
echo "E2E CPU ORT DONE"
