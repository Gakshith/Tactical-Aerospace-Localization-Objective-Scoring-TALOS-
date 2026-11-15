#!/usr/bin/env bash
#SBATCH --job-name=talos-export
#SBATCH --partition=gpuquick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
# Day 5: export the chosen detector (y26s_t1024) to ONNX / TensorRT, then run every
# runtime over AIR-MOT val on this one GPU and score each with det_metrics, so speed and
# accuracy come from the same frames and the same node. Submit from the repo root:
#   cd ~/talos/repo && sbatch jobs/export_parity.sbatch
# Env: SRC_PT (default: configs/talos_v2.json weights), SPLITS (val), SKIP_EXPORT=1,
#      ONLY ("name name ..." = run just these variants).
# ONNX Runtime on the GPU runs from ~/talos/.venv-ort: the onnxruntime-gpu 1.31 wheel is built for
# CUDA 13 (+ tensorrt-cu13), while the main venv is CUDA 12 (torch cu128, tensorrt-cu12). That
# path imports no torch, so it needs no CUDA 12 libraries.
source "$HOME/talos/repo/jobs/common.sh"

SRC_PT="${SRC_PT:-$TALOS/runs/det/y26s_t1024/weights/best.pt}"
EXP="$WEIGHTS/export"
STEM="$EXP/y26s_t1024"
DETS="$DATA/dets"
OUT="results/speed"
mkdir -p "$EXP" "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ "${SKIP_EXPORT:-0}" != 1 ]; then
  cp "$SRC_PT" "$STEM.pt"
  python -m src.TALOS.detect.export_onnx --weights "$STEM.pt" --imgsz 1024 \
    --onnx --fuse-preprocess --int8-dynamic --engine-half --engine-batch 64
  python - "$EXP/export_manifest.json" "$OUT/export_manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
json.dump(m, open(sys.argv[2], "w"), indent=2)   # file names + sha256 only, no paths
PY
fi

# name | backend | weights | extra args
VARIANTS=(
  "rt_torch_fp32|torch|$STEM.pt|"
  "rt_torch_fp16|torch|$STEM.pt|--half"
  "rt_ort_cuda_fp32|onnx|${STEM}_u8.onnx|--ort-provider cuda"
  "rt_ort_trt_fp16|onnx|${STEM}_u8.onnx|--ort-provider tensorrt --half"
  "rt_trt_fp16|trt|$STEM.engine|"
)
ORT_VENV="$TALOS/.venv-ort"
ORT_SP=$("$ORT_VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")
ORT_LD="$ORT_SP/tensorrt_libs:$(ls -d "$ORT_SP"/nvidia/*/lib | tr '\n' ':')"
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r name backend w extra <<<"$v"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  PY=python; ENVV=()
  if [ "$backend" = onnx ]; then PY="$ORT_VENV/bin/python"; ENVV=(env "LD_LIBRARY_PATH=$ORT_LD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"); fi
  echo "=== $name ($backend, $(basename "$w"))"
  rm -rf "$DETS/$name"
  for split in ${SPLITS:-val}; do
    # shellcheck disable=SC2086
    "${ENVV[@]}" "$PY" -m src.TALOS.detect.run_detector --weights "$w" --name "$name" --mode tiled --tile 1024 \
      --overlap 0.2 --imgsz 1024 --classes 0:AIRMOT_plane,1:AIRMOT_ship --mot-root "$DATA/mot" \
      --split "$split" --conf 0.1 --v1-conf 0.25 --end2end true --batch 64 --max-det 300 \
      --backend "$backend" --out-root "$DETS" $extra
  done
  for b in plane ship; do
    python -m src.TALOS.eval.det_metrics --dets-root "$DETS" --detector "$name" --bench "AIRMOT_$b" \
      --bench-dir "$DATA/mot/AIRMOT_$b-all" --splits ${SPLITS:-val} --conf 0.25 --out "$OUT/$name/$b.json"
  done
  cp "$DETS/$name/meta.json" "$OUT/$name/meta.json"
done
python scripts/speed_table.py --root "$OUT" --ref rt_torch_fp32 --out "$OUT/detector_speed.json"
echo "EXPORT+PARITY DONE"
