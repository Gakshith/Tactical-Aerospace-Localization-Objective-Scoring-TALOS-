#!/usr/bin/env bash
#SBATCH --job-name=talos-smoke
#SBATCH --partition=gpuquick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm-%j.out
# Slurm expands neither ~ nor $HOME here; a relative path resolves against the
# submit directory, so always submit from the repo root:
#   cd ~/talos/repo && sbatch jobs/smoke_gpu.sbatch

source "$HOME/talos/repo/jobs/common.sh"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

python - <<'PY'
import numpy as np, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
assert torch.cuda.is_available(), "CUDA not available"
name = torch.cuda.get_device_name(0)
print("device:", name)
assert "A100" in name, name
x = torch.randn(4096, 4096, device="cuda")
print("matmul ok:", float((x @ x).sum().isfinite()))

import os, ultralytics
from ultralytics import YOLO
print("ultralytics", ultralytics.__version__)
model = YOLO(os.path.join(os.environ["WEIGHTS"], "yolo26s.pt"))
img = (np.random.default_rng(0).integers(0, 255, (1024, 1024, 3))).astype(np.uint8)
res = model.predict(img, imgsz=1024, device="cuda:0", verbose=False)
print("predict ok: boxes", len(res[0].boxes), "speed(ms)", {k: round(v, 1) for k, v in res[0].speed.items()})
print("predictor device:", model.predictor.device)
assert model.predictor.device.type == "cuda"

import trackeval
from trackeval.metrics import HOTA, CLEAR, Identity
# toy sequence: 2 GT ids tracked perfectly for 3 frames -> HOTA = MOTA = 1
T = 3
data = dict(num_timesteps=T, num_gt_ids=2, num_tracker_ids=2, num_gt_dets=2*T, num_tracker_dets=2*T,
            gt_ids=[np.array([0, 1])]*T, tracker_ids=[np.array([0, 1])]*T,
            similarity_scores=[np.eye(2)]*T)
hota = HOTA().eval_sequence(data)["HOTA"].mean()
mota = CLEAR().eval_sequence(data)["MOTA"]
print(f"trackeval ok: toy HOTA={hota:.3f} MOTA={mota:.3f}")
assert abs(hota - 1) < 1e-6 and abs(mota - 1) < 1e-6

import trackers, supervision as sv
names = sorted(n for n in dir(trackers) if n.endswith("Tracker"))
print("trackers", getattr(trackers, "__version__", "?"), "classes:", names)
dets = sv.Detections(xyxy=np.array([[10, 10, 50, 50], [100, 100, 160, 180]], float),
                     confidence=np.array([0.9, 0.8]), class_id=np.array([0, 0]))
for n in ["SORTTracker", "ByteTrackTracker", "OCSORTTracker", "BoTSORTTracker"]:
    t = getattr(trackers, n)()
    for _ in range(3):
        out = t.update(dets)
    print(f"  {n}: tracker_ids={list(out.tracker_id)}")

import onnxruntime as ort
print("onnxruntime", ort.__version__, ort.get_available_providers())
import sahi, cv2
print("sahi", sahi.__version__, "cv2", cv2.__version__)
print("SMOKE PASSED")
PY
