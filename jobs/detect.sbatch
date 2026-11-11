#!/usr/bin/env bash
#SBATCH --job-name=talos-detect
#SBATCH --partition=gpuquick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
# Fill the detection cache for one benchmark, one or more splits.
# Submit from the repo root (relative --output):
#   cd ~/talos/repo && NAME=v1_cgi WEIGHTS_PT=$HOME/talos/weights/v1_cgi_best.pt \
#     BENCH=AIRMOT_plane SPLITS="val test train" sbatch --export=ALL jobs/detect.sbatch
# Env: WEIGHTS_PT, NAME, BENCH, SPLITS (space-separated), CONF (save, 0.1),
#      V1_CONF (operating, 0.5), IMGSZ (empty = checkpoint default), BATCH (16),
#      BENCH_DIR (default $DATA/mot/<BENCH>-all), OUT_ROOT (default $DATA/dets),
#      END2END (auto|true|false; v1 needs true: ultralytics 8.4.210 otherwise runs
#      the YOLO26 checkpoint's one2many+NMS head, not the end2end head v1 used).
# (WEIGHTS is taken by common.sh for the weights directory, hence WEIGHTS_PT.)

source "$HOME/talos/repo/jobs/common.sh"

: "${WEIGHTS_PT:?set WEIGHTS_PT}" "${NAME:?set NAME}" "${BENCH:?set BENCH}"
SPLITS="${SPLITS:-val test}"
BENCH_DIR="${BENCH_DIR:-$DATA/mot/${BENCH}-all}"
OUT_ROOT="${OUT_ROOT:-$DATA/dets}"
IMGSZ_ARG=()
[ -n "${IMGSZ:-}" ] && IMGSZ_ARG=(--imgsz "$IMGSZ")

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for split in $SPLITS; do
  python -m src.TALOS.detect.run_detector \
    --weights "$WEIGHTS_PT" --name "$NAME" \
    --bench-dir "$BENCH_DIR" --bench "$BENCH" --split "$split" \
    --conf "${CONF:-0.1}" --v1-conf "${V1_CONF:-0.5}" "${IMGSZ_ARG[@]}" \
    --out-root "$OUT_ROOT" --batch "${BATCH:-16}" --end2end "${END2END:-auto}"
done
echo "DETECT DONE"
