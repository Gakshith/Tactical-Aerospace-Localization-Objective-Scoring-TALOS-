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
# Tiled / multi-class (one cache per bench, each sequence run once):
#   NAME=coco_y26s_tiled WEIGHTS_PT=$HOME/talos/weights/yolo26s.pt MODE=tiled TILE=640 OVERLAP=0.2 \
#     CLASSES=4:AIRMOT_plane,8:AIRMOT_ship SPLITS=val END2END=true sbatch --export=ALL jobs/detect.sbatch
# Env: MODE (full|tiled, default full), TILE (640), OVERLAP (0.2), WITH_FULL_FRAME (1 = add a
#      downscaled whole-frame pass), FULL_IMGSZ, MAX_DET (300), CLASSES (id:bench,...; replaces
#      BENCH and uses MOT_ROOT, default $DATA/mot). In tiled mode BATCH counts tiles (default 64).

source "$HOME/talos/repo/jobs/common.sh"

: "${WEIGHTS_PT:?set WEIGHTS_PT}" "${NAME:?set NAME}"
SPLITS="${SPLITS:-val test}"
MODE="${MODE:-full}"
OUT_ROOT="${OUT_ROOT:-$DATA/dets}"
IMGSZ_ARG=()
[ -n "${IMGSZ:-}" ] && IMGSZ_ARG=(--imgsz "$IMGSZ")
if [ -n "${CLASSES:-}" ]; then
  TARGET_ARGS=(--classes "$CLASSES" --mot-root "${MOT_ROOT:-$DATA/mot}")
else
  : "${BENCH:?set BENCH or CLASSES}"
  TARGET_ARGS=(--bench "$BENCH" --bench-dir "${BENCH_DIR:-$DATA/mot/${BENCH}-all}" --cls "${CLS:-0}")
fi
MODE_ARGS=(--mode "$MODE" --max-det "${MAX_DET:-300}")
if [ "$MODE" = tiled ]; then
  MODE_ARGS+=(--tile "${TILE:-640}" --overlap "${OVERLAP:-0.2}")
  [ "${WITH_FULL_FRAME:-0}" = 1 ] && MODE_ARGS+=(--with-full-frame)
  [ -n "${FULL_IMGSZ:-}" ] && MODE_ARGS+=(--full-imgsz "$FULL_IMGSZ")
  BATCH="${BATCH:-64}"
fi
HALF_ARG=()
[ "${HALF:-0}" = 1 ] && HALF_ARG=(--half)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for split in $SPLITS; do
  python -m src.TALOS.detect.run_detector \
    --weights "$WEIGHTS_PT" --name "$NAME" "${TARGET_ARGS[@]}" "${MODE_ARGS[@]}" --split "$split" \
    --conf "${CONF:-0.1}" --v1-conf "${V1_CONF:-0.5}" "${IMGSZ_ARG[@]}" "${HALF_ARG[@]}" \
    --out-root "$OUT_ROOT" --batch "${BATCH:-16}" --end2end "${END2END:-auto}"
done
echo "DETECT DONE"
