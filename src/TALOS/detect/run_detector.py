"""Run an Ultralytics detector over MOT benchmark sequences and write the
detection cache (see ``cache.py``).

    python -m src.TALOS.detect.run_detector --weights W --name v1_cgi \
        --bench-dir ~/talos/data/mot/AIRMOT_plane-all --bench AIRMOT_plane \
        --split val --conf 0.1 --v1-conf 0.5 --out-root ~/talos/data/dets

Frame index = 1-based position in the sorted list of ``img1/*<imExt>`` files,
which must equal ``seqLength``. Boxes are saved at ``--conf`` (low, so later
trackers can use weak boxes); the detector's own operating threshold is
recorded in meta as ``v1_conf`` and applied at evaluation time.

``--imgsz`` is optional: when omitted, Ultralytics uses the checkpoint's
training imgsz (that is how v1 ran). The imgsz actually used is read back from
the predictor and recorded.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

from src.TALOS.data.mot import read_seqinfo
from src.TALOS.detect import cache


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_paths(seq_dir: Path) -> list[Path]:
    info = read_seqinfo(seq_dir)
    frames = sorted((seq_dir / info.img_dir).glob(f"*{info.img_ext}"))
    if len(frames) != info.length:
        raise RuntimeError(f"{seq_dir.name}: {len(frames)} *{info.img_ext} frames, seqLength={info.length}")
    return frames


def resolve_seqs(a) -> list[str]:
    present = sorted(p.name for p in a.bench_dir.iterdir() if (p / "seqinfo.ini").exists())
    if a.seqs:
        missing = set(a.seqs) - set(present)
        if missing:
            raise SystemExit(f"not in {a.bench_dir}: {sorted(missing)}")
        return sorted(a.seqs)
    split = json.loads(Path(a.split_file).read_text())["split"][a.split]
    # the split covers every annotated video; a per-class bench holds only those with that class
    return [s for s in sorted(split) if s in present]


@contextmanager
def locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--name", required=True, help="detector name in the cache")
    ap.add_argument("--bench-dir", type=Path, required=True, help="dir holding <seq>/seqinfo.ini")
    ap.add_argument("--bench", required=True, help="bench name in the cache, e.g. AIRMOT_plane")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seqs", nargs="+")
    g.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--split-file", default="configs/splits/airmot.json")
    ap.add_argument("--conf", type=float, default=0.1, help="save threshold")
    ap.add_argument("--v1-conf", type=float, default=0.5, help="operating threshold recorded in meta")
    ap.add_argument("--imgsz", type=int, default=None, help="omit to use the checkpoint default")
    ap.add_argument("--cls", type=int, default=0, help="class id to keep")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--half", action="store_true", help="fp16 (v1 ran fp32)")
    ap.add_argument("--end2end", choices=["auto", "true", "false"], default="auto",
                    help="YOLO26 head. ultralytics 8.4.210 loads v1's checkpoint with end2end=False "
                         "(one2many head + NMS); v1 ran on 8.4.14, which used the end2end head, so "
                         "reproducing v1 needs 'true'")
    a = ap.parse_args()

    from ultralytics import YOLO
    import ultralytics

    assert torch.cuda.is_available(), "CUDA not available"
    device = "cuda:0"  # the one GPU Slurm gave us
    seqs = resolve_seqs(a)
    print(f"[detect] {a.name} on {a.bench} {a.split or 'seqs'}: {len(seqs)} seqs", flush=True)

    model = YOLO(str(a.weights))
    kw = dict(conf=a.conf, device=device, verbose=False, half=a.half)
    if a.imgsz:
        kw["imgsz"] = a.imgsz
    if a.end2end != "auto":
        kw["end2end"] = a.end2end == "true"

    # warm-up (cudnn autotune, lazy init) so timing reflects steady state
    first = cv2.imread(str(frame_paths(a.bench_dir / seqs[0])[0]))
    for _ in range(3):
        model.predict([first] * a.batch, **kw)
    torch.cuda.synchronize()
    imgsz_used = model.predictor.imgsz  # after check_imgsz; args.imgsz is the raw value
    if isinstance(imgsz_used, (list, tuple)):
        imgsz_used = [int(v) for v in imgsz_used]
    # the override is consumed by the model, not kept in predictor.args, and the head's own
    # attribute stays False on 8.4.210; record the value we passed (None = library default)
    head_e2e = kw.get("end2end")
    print(f"[detect] imgsz used: {imgsz_used}  head end2end: {head_e2e}  "
          f"(ckpt overrides: {model.overrides})", flush=True)

    per_seq = {}
    tot_frames, t_pred, t_all = 0, 0.0, 0.0
    us = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    t_job = time.perf_counter()
    for seq in seqs:
        frames = frame_paths(a.bench_dir / seq)
        rows = []
        t0 = time.perf_counter()
        sp = 0.0
        for s in range(0, len(frames), a.batch):
            imgs = [cv2.imread(str(p)) for p in frames[s:s + a.batch]]
            if any(im is None for im in imgs):
                raise FileNotFoundError(frames[s:s + a.batch])
            torch.cuda.synchronize()
            tp = time.perf_counter()
            res = model.predict(imgs, **kw)
            torch.cuda.synchronize()
            sp += time.perf_counter() - tp
            for j, r in enumerate(res):
                for k in us:
                    us[k] += r.speed[k]
                b = r.boxes
                if b is None or len(b) == 0:
                    continue
                keep = (b.cls == a.cls).cpu().numpy()
                xyxy = b.xyxy.cpu().numpy()[keep]
                cf = b.conf.cpu().numpy()[keep]
                f = s + j + 1
                rows.append(np.column_stack([np.full(len(cf), f), xyxy, cf]))
        dt = time.perf_counter() - t0
        dets = np.concatenate(rows) if rows else np.zeros((0, 6), np.float32)
        p = cache.save(a.out_root, a.name, a.bench, seq, dets)
        n05 = int((dets[:, 5] >= a.v1_conf).sum()) if len(dets) else 0
        per_seq[seq] = {"frames": len(frames), "dets": len(dets), f"dets_ge_{a.v1_conf:g}": n05,
                        "predict_ms_per_frame": round(1000 * sp / len(frames), 3),
                        "e2e_ms_per_frame": round(1000 * dt / len(frames), 3)}
        tot_frames += len(frames)
        t_pred += sp
        t_all += dt
        print(f"[detect] {seq}: {len(frames)} frames, {len(dets)} dets (>= {a.v1_conf:g}: {n05}), "
              f"{1000 * sp / len(frames):.1f} ms/f predict, {1000 * dt / len(frames):.1f} ms/f e2e -> {p}", flush=True)

    run = {
        "bench": a.bench, "bench_dir": str(a.bench_dir), "split": a.split, "seqs": seqs,
        "total_frames": tot_frames, "wall_time_s": round(time.perf_counter() - t_job, 2),
        "batch": a.batch, "half": a.half,
        "mean_ms_per_frame_predict": round(1000 * t_pred / max(tot_frames, 1), 3),
        "mean_ms_per_frame_e2e": round(1000 * t_all / max(tot_frames, 1), 3),
        "ultralytics_speed_ms_per_frame": {k: round(v / max(tot_frames, 1), 3) for k, v in us.items()},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "per_seq": per_seq,
    }
    common = {
        "detector": a.name,
        "weights": str(a.weights.resolve()), "weights_sha256": sha256(a.weights),
        "save_conf": a.conf, "v1_conf": a.v1_conf, "class_kept": a.cls,
        "imgsz_arg": a.imgsz, "imgsz_used": imgsz_used,
        "end2end_arg": a.end2end, "head_end2end_used": head_e2e,
        "predict_kwargs": {k: v for k, v in kw.items() if k != "device"},
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "frame_index": "1-based position in sorted img1/*<imExt>, == seqLength",
        "cache_format": "<root>/<detector>/<bench>/<seq>.npy, (N,6) [frame,x1,y1,x2,y2,conf]",
    }
    meta_path = a.out_root / a.name / "meta.json"
    with locked(a.out_root / a.name / ".meta.lock"):
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        old = {k: meta.get(k) for k in ("weights_sha256", "save_conf", "imgsz_used", "head_end2end_used")}
        if meta and any(old[k] != common[k] for k in old):
            raise SystemExit(f"meta.json disagrees with this run: {old}")
        meta.update(common)
        meta.setdefault("runs", {})[f"{a.bench}/{a.split or 'seqs'}"] = run
        cache.write_meta(a.out_root, a.name, meta)
    print(f"[detect] done: {tot_frames} frames, predict {run['mean_ms_per_frame_predict']} ms/f, "
          f"e2e {run['mean_ms_per_frame_e2e']} ms/f, meta -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()
