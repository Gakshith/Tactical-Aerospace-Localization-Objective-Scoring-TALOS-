#!/usr/bin/env bash
#SBATCH --job-name=talos-train-det
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm-%j.out
# Fine-tune a YOLO26 detector on a tiled AIR-MOT set (src/TALOS/data/make_yolo_tiles.py).
# Submit from the repo root (relative --output):
#   cd ~/talos/repo && NAME=y26s_t640 MODEL=yolo26s.pt IMGSZ=640 BATCH=64 \
#     DATA_YAML=$HOME/talos/data/yolo/airmot_t640/data.yaml sbatch --export=ALL jobs/train_det.sbatch
# Env: DATA_YAML, NAME (required); MODEL (file in $WEIGHTS or a path, yolo26s.pt),
#      IMGSZ (640), EPOCHS (100), BATCH (32), PATIENCE (20), WORKERS (12),
#      DEGREES (0), SCALE (0.3), CACHE (false|ram|disk), SEED (0).
# Output: ~/talos/runs/det/<NAME>/  (weights/best.pt, results.csv, args.yaml,
#         best_summary.json, val_e2e/ and gpu_mem.csv).
#
# YOLO26 head: ultralytics 8.4.210 validates with the one2many head + NMS unless
# nms=False (the old end2end=True). We pass nms=False to train, so the per-epoch
# val in results.csv, early stopping and best.pt selection all use the end2end
# head, and best.pt is saved with end2end on. A separate end2end val of best.pt
# at the end writes best_summary.json (per-class P/R/mAP); the one2many+NMS
# numbers are recorded beside it only as a cross-check.
#
# Augmentation: flipud + fliplr 0.5 (with both, a tile also gets the 180° turn),
# mosaic on (off for the last 10 epochs). degrees stays 0 by default: Ultralytics
# rotates an axis-aligned box by its corners, inflating it (x1.41 at 45°), which
# costs more on 10 px ships than the extra orientations gain. scale 0.3 (default
# 0.5 would shrink 10 px ships to 5 px).

source "$HOME/talos/repo/jobs/common.sh"

: "${DATA_YAML:?set DATA_YAML}" "${NAME:?set NAME}"
MODEL="${MODEL:-yolo26s.pt}"
[ -f "$MODEL" ] || MODEL="$WEIGHTS/$MODEL"
export DATA_YAML NAME MODEL
export IMGSZ="${IMGSZ:-640}" EPOCHS="${EPOCHS:-100}" BATCH="${BATCH:-32}" PATIENCE="${PATIENCE:-20}"
export WORKERS="${WORKERS:-12}" DEGREES="${DEGREES:-0}" SCALE="${SCALE:-0.3}" CACHE="${CACHE:-false}"
export SEED="${SEED:-0}"
export PROJECT="$TALOS/runs/det"
RUN_DIR="$PROJECT/$NAME"
[ -e "$RUN_DIR" ] && { echo "refusing: $RUN_DIR exists"; exit 1; }
mkdir -p "$RUN_DIR"
cp "$REPO/.talos_source" "$RUN_DIR/talos_source.txt" 2>/dev/null || true

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# GPU memory trace for this job's GPU (CUDA_VISIBLE_DEVICES is set by Slurm)
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=timestamp,memory.used,utilization.gpu \
  --format=csv,noheader -l 60 > "$RUN_DIR/gpu_mem.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

python - <<'PY'
import csv, json, os, time
from pathlib import Path

import torch
from ultralytics import YOLO

E = os.environ
name, project = E["NAME"], E["PROJECT"]
run = Path(project) / name
cache = {"false": False, "ram": "ram", "disk": "disk"}[E["CACHE"].lower()]
t0 = time.time()
model = YOLO(E["MODEL"])
model.train(
    data=E["DATA_YAML"], imgsz=int(E["IMGSZ"]), epochs=int(E["EPOCHS"]), batch=int(E["BATCH"]),
    patience=int(E["PATIENCE"]), workers=int(E["WORKERS"]), seed=int(E["SEED"]), deterministic=True,
    project=project, name=name, exist_ok=True, cache=cache,
    nms=False,  # end2end head for every val during training (see header)
    flipud=0.5, fliplr=0.5, degrees=float(E["DEGREES"]), scale=float(E["SCALE"]),
    mosaic=1.0, close_mosaic=10, plots=True, verbose=True,
)
train_s = time.time() - t0
peak_reserved = torch.cuda.max_memory_reserved() / 2**30

rows = list(csv.DictReader(open(run / "results.csv")))
rows = [{k.strip(): v for k, v in r.items()} for r in rows]
key = "metrics/mAP50-95(B)"  # ultralytics fitness for detect = mAP50-95
best = max(rows, key=lambda r: float(r[key]))

def per_class(res):
    names = res.names
    out = {}
    for i, c in enumerate(res.box.ap_class_index):
        out[names[int(c)]] = {"P": float(res.box.p[i]), "R": float(res.box.r[i]),
                              "mAP50": float(res.box.ap50[i]), "mAP50-95": float(res.box.ap[i])}
    out["all"] = {"P": float(res.box.mp), "R": float(res.box.mr),
                  "mAP50": float(res.box.map50), "mAP50-95": float(res.box.map)}
    return out

best_pt = run / "weights" / "best.pt"
common = dict(data=E["DATA_YAML"], imgsz=int(E["IMGSZ"]), batch=int(E["BATCH"]), split="val",
              plots=False, project=str(run), exist_ok=True, verbose=True)
e2e = YOLO(str(best_pt)).val(nms=False, name="val_e2e", **common)
o2m = YOLO(str(best_pt)).val(nms=None, name="val_o2m_nms", **common)

summary = {
    "name": name, "model": E["MODEL"], "data": E["DATA_YAML"], "imgsz": int(E["IMGSZ"]),
    "batch": int(E["BATCH"]), "epochs_max": int(E["EPOCHS"]), "patience": int(E["PATIENCE"]),
    "epochs_run": len(rows), "best_epoch": int(float(best["epoch"])),
    "best_epoch_row": best, "train_wall_s": round(train_s, 1),
    "train_wall_h": round(train_s / 3600, 2), "peak_cuda_reserved_gib": round(peak_reserved, 2),
    "gpu": torch.cuda.get_device_name(0),
    "weights": str(best_pt),
    "val_end2end": per_class(e2e),
    "val_one2many_nms_crosscheck": per_class(o2m),
    "note": "tile-level val (val split tiles), end2end head (nms=False); test split never read",
}
(run / "best_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in ("epochs_run", "best_epoch", "train_wall_h",
                                          "peak_cuda_reserved_gib", "val_end2end")}, indent=2))
PY
echo "TRAIN DONE"
