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

Modes. ``--mode full`` (default) runs the detector on the whole frame.
``--mode tiled`` runs sliced inference (``tiled.py``): ``--tile`` px tiles at
``--overlap``, native resolution (``--imgsz`` defaults to ``--tile``), merged
back to frame pixels; ``--with-full-frame`` adds one downscaled whole-frame pass
at ``--full-imgsz``. In tiled mode ``--batch`` counts tiles per GPU call.

Several classes in one pass. ``--classes 0:AIRMOT_plane,1:AIRMOT_ship`` maps
model class ids to bench names (COCO: ``4:AIRMOT_plane,8:AIRMOT_ship``) and
writes one cache per bench; bench dirs are ``--mot-root/<bench>-all``. Each
sequence present in any bench is run once. Without ``--classes`` the old
single-bench flags (``--bench --bench-dir --cls``) behave as before.

    python -m src.TALOS.detect.run_detector --weights ~/talos/weights/yolo26s.pt \
        --name coco_y26s_tiled --mode tiled --tile 640 --overlap 0.2 \
        --classes 4:AIRMOT_plane,8:AIRMOT_ship --mot-root ~/talos/data/mot \
        --split val --end2end true --out-root ~/talos/data/dets

Backends (tiled mode). ``--backend torch`` (default) runs Ultralytics on the .pt;
``--backend onnx`` runs ONNX Runtime on an end2end .onnx (``--ort-provider cpu |
cuda | tensorrt``; ``--half`` = TensorRT fp16; ``--threads`` = CPU intra-op
threads) and needs neither torch nor ultralytics; ``--backend trt`` runs
Ultralytics on a TensorRT .engine. ``--weights`` names the artifact. ``--device
cpu`` runs without a GPU (torch or onnx).
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

from src.TALOS.data.mot import read_seqinfo
from src.TALOS.detect import cache
from src.TALOS.detect.tiled import BACKENDS, TiledDetector, UltralyticsBackend

torch = None  # imported only when a torch-based backend or the GPU is used


def _sync() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


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


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor() or None


def _gpu_name() -> str | None:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i",
                              os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out.splitlines()[0] if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_seqs(a, bench_dir: Path, strict: bool = True) -> list[str]:
    present = sorted(p.name for p in bench_dir.iterdir() if (p / "seqinfo.ini").exists())
    if a.seqs:
        missing = set(a.seqs) - set(present)
        if missing and strict:
            raise SystemExit(f"not in {bench_dir}: {sorted(missing)}")
        return sorted(set(a.seqs) & set(present))
    split = json.loads(Path(a.split_file).read_text())["split"][a.split]
    # the split covers every annotated video; a per-class bench holds only those with that class
    return [s for s in sorted(split) if s in present]


def parse_classes(spec: str) -> dict[int, str]:
    out = {}
    for item in spec.split(","):
        cid, bench = item.split(":")
        out[int(cid)] = bench.strip()
    if len(set(out.values())) != len(out):
        raise SystemExit(f"--classes maps two ids to one bench: {spec}")
    return out


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
    ap.add_argument("--bench-dir", type=Path, help="dir holding <seq>/seqinfo.ini (single-bench mode)")
    ap.add_argument("--bench", help="bench name in the cache, e.g. AIRMOT_plane (single-bench mode)")
    ap.add_argument("--classes", help="model class id -> bench, e.g. 4:AIRMOT_plane,8:AIRMOT_ship; "
                                      "one cache per bench (replaces --bench/--cls)")
    ap.add_argument("--mot-root", type=Path, help="with --classes: bench dirs are <mot-root>/<bench>-all")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seqs", nargs="+")
    g.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--split-file", default="configs/splits/airmot.json")
    ap.add_argument("--conf", type=float, default=0.1, help="save threshold")
    ap.add_argument("--v1-conf", type=float, default=0.5, help="operating threshold recorded in meta")
    ap.add_argument("--imgsz", type=int, default=None, help="omit to use the checkpoint default")
    ap.add_argument("--cls", type=int, default=0, help="class id to keep (single-bench mode)")
    ap.add_argument("--mode", choices=["full", "tiled"], default="full")
    ap.add_argument("--tile", type=int, default=640, help="tiled: tile size, px")
    ap.add_argument("--overlap", type=float, default=0.2, help="tiled: overlap fraction between tiles")
    ap.add_argument("--with-full-frame", action="store_true", help="tiled: add a downscaled whole-frame pass")
    ap.add_argument("--full-imgsz", type=int, default=None, help="tiled: imgsz of that pass (default = imgsz)")
    ap.add_argument("--max-det", type=int, default=300, help="max boxes per image (per tile in tiled mode)")
    ap.add_argument("--gpu-tiling", action="store_true",
                    help="tiled: crop tiles on the GPU and pass a tensor (no net speed gain measured; default off)")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16, help="frames per call (full) / tiles per call (tiled)")
    ap.add_argument("--half", action="store_true", help="fp16 (v1 ran fp32); onnx: TensorRT fp16")
    ap.add_argument("--backend", choices=BACKENDS, default="torch",
                    help="tiled: torch (.pt) | onnx (ONNX Runtime, .onnx) | trt (Ultralytics TensorRT .engine)")
    ap.add_argument("--ort-provider", choices=["cpu", "cuda", "tensorrt"], default=None,
                    help="onnx backend execution provider (default: cuda on --device cuda, else cpu)")
    ap.add_argument("--threads", type=int, default=None, help="onnx on CPU: intra-op threads")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 (the one GPU Slurm gave us) or cpu")
    ap.add_argument("--end2end", choices=["auto", "true", "false"], default="auto",
                    help="YOLO26 head. ultralytics 8.4.210 loads v1's checkpoint with end2end=False "
                         "(one2many head + NMS); v1 ran on 8.4.14, which used the end2end head, so "
                         "reproducing v1 needs 'true'")
    a = ap.parse_args()

    global torch
    if a.backend != "torch" and a.mode != "tiled":
        raise SystemExit("--backend onnx/trt needs --mode tiled")
    device = a.device
    ort_cpu = a.backend == "onnx" and (a.ort_provider == "cpu" or (a.ort_provider is None and device == "cpu"))
    if a.backend != "onnx":  # the onnx backend runs without torch (ORT calls are synchronous)
        import torch as _t
        torch = _t
    if device.startswith("cuda") and a.backend != "onnx":
        assert torch.cuda.is_available(), "CUDA not available"
    try:
        import ultralytics
        ul_version = ultralytics.__version__
    except ImportError:
        if a.backend != "onnx":
            raise
        ul_version = None

    # ---- which classes go to which bench, and where each bench lives
    if a.classes:
        class_map = parse_classes(a.classes)
        if a.mot_root is None and not (a.bench_dir and len(class_map) == 1):
            raise SystemExit("--classes needs --mot-root (or one class and --bench-dir)")
        bench_dirs = {b: (a.mot_root / f"{b}-all" if a.mot_root else a.bench_dir) for b in class_map.values()}
    else:
        if not (a.bench and a.bench_dir):
            raise SystemExit("give --bench and --bench-dir, or --classes and --mot-root")
        class_map = {a.cls: a.bench}
        bench_dirs = {a.bench: a.bench_dir}
    bench_of = class_map  # model class id -> bench
    bench_seqs = {b: resolve_seqs(a, d, strict=len(bench_dirs) == 1) for b, d in bench_dirs.items()}
    if a.seqs:
        missing = set(a.seqs) - set().union(*bench_seqs.values())
        if missing:
            raise SystemExit(f"in no bench: {sorted(missing)}")
    seqs = sorted(set().union(*bench_seqs.values()))
    # one frame source per sequence; benches share the images (img1 symlinks to the same raw video)
    seq_dir = {}
    for s in seqs:
        dirs = [bench_dirs[b] / s for b in bench_dirs if s in bench_seqs[b]]
        imgs = {(d / read_seqinfo(d).img_dir).resolve() for d in dirs}
        if len(imgs) != 1:
            raise SystemExit(f"{s}: benches point at different images {sorted(map(str, imgs))}")
        seq_dir[s] = dirs[0]
    print(f"[detect] {a.name} ({a.mode}) {a.split or 'seqs'}: {len(seqs)} seqs; "
          + ", ".join(f"cls {c}->{b} ({len(bench_seqs[b])} seqs)" for c, b in bench_of.items()), flush=True)

    e2e = None if a.end2end == "auto" else a.end2end == "true"
    if a.mode == "full":
        det = UltralyticsBackend(a.weights, imgsz=a.imgsz, conf=a.conf, end2end=e2e, device=device,
                                 half=a.half, classes=sorted(bench_of), max_det=a.max_det)
        frames_per_call = a.batch
    else:
        det = TiledDetector(a.weights, tile=a.tile, overlap=a.overlap, imgsz=a.imgsz, conf=a.conf, end2end=e2e,
                            device=device, batch=a.batch, with_full_frame=a.with_full_frame,
                            full_imgsz=a.full_imgsz, half=a.half, classes=sorted(bench_of), max_det=a.max_det,
                            gpu_tiling=a.gpu_tiling, backend=a.backend, ort_provider=a.ort_provider,
                            threads=a.threads)
        info0 = read_seqinfo(seq_dir[seqs[0]])
        n_tiles = len(det.grid(info0.width, info0.height))
        frames_per_call = max(1, -(-a.batch // n_tiles))

    # warm-up (cudnn autotune, lazy init) so timing reflects steady state
    first = cv2.imread(str(frame_paths(seq_dir[seqs[0]])[0]))
    for _ in range(3):
        det.predict([first] * frames_per_call) if a.mode == "tiled" else det([first] * frames_per_call)
    _sync()
    desc = det.describe()
    if a.mode == "tiled":
        det.timing = {k: 0 * v for k, v in det.timing.items()}
        for bk in (det.backend, det.full_backend):
            if hasattr(bk, "speed_ms"):
                bk.speed_ms = {k: 0.0 for k in bk.speed_ms}
                bk.images_seen = 0
        tb = desc["tile_backend"]
        imgsz_used, head_e2e = tb["imgsz_used"], tb["head_end2end_active"]
    else:
        det.speed_ms = {k: 0.0 for k in det.speed_ms}
        det.images_seen = 0
        imgsz_used, head_e2e = desc["imgsz_used"], desc["head_end2end_active"]
    if e2e is None and head_e2e is False:
        print("[detect] WARNING: --end2end auto ran a NMS head; YOLO26 checkpoints on ultralytics 8.4.210 "
              "need --end2end true", flush=True)
    print(f"[detect] imgsz used: {imgsz_used}  end2end arg: {e2e}  head end2end active: {head_e2e}  "
          f"frames/call: {frames_per_call}", flush=True)

    per_seq = {}
    tot_frames, t_pred, t_all = 0, 0.0, 0.0
    t_job = time.perf_counter()
    for seq in seqs:
        frames = frame_paths(seq_dir[seq])
        rows = {b: [] for b in bench_dirs}
        t0 = time.perf_counter()
        sp = 0.0
        for s in range(0, len(frames), frames_per_call):
            imgs = [cv2.imread(str(p)) for p in frames[s:s + frames_per_call]]
            if any(im is None for im in imgs):
                raise FileNotFoundError(frames[s:s + frames_per_call])
            _sync()
            tp = time.perf_counter()
            res = det.predict(imgs) if a.mode == "tiled" else det(imgs)
            _sync()
            sp += time.perf_counter() - tp
            for j, d in enumerate(res):
                if len(d) == 0:
                    continue
                f = s + j + 1
                for cid, b in bench_of.items():
                    m = d[:, 5] == cid
                    if m.any():
                        rows[b].append(np.column_stack([np.full(int(m.sum()), f), d[m, :5]]))
        dt = time.perf_counter() - t0
        per_seq[seq] = {"frames": len(frames), "predict_ms_per_frame": round(1000 * sp / len(frames), 3),
                        "e2e_ms_per_frame": round(1000 * dt / len(frames), 3)}
        for b in bench_dirs:
            if seq not in bench_seqs[b]:
                continue
            dets = np.concatenate(rows[b]) if rows[b] else np.zeros((0, 6), np.float32)
            p = cache.save(a.out_root, a.name, b, seq, dets)
            n05 = int((dets[:, 5] >= a.v1_conf).sum()) if len(dets) else 0
            per_seq[seq][b] = {"dets": len(dets), f"dets_ge_{a.v1_conf:g}": n05}
            print(f"[detect] {seq} {b}: {len(frames)} frames, {len(dets)} dets (>= {a.v1_conf:g}: {n05}) -> {p}",
                  flush=True)
        tot_frames += len(frames)
        t_pred += sp
        t_all += dt
        print(f"[detect] {seq}: {1000 * sp / len(frames):.1f} ms/f predict, {1000 * dt / len(frames):.1f} ms/f e2e",
              flush=True)

    nf = max(tot_frames, 1)
    timing = {
        "total_frames": tot_frames, "wall_time_s": round(time.perf_counter() - t_job, 2),
        "batch": a.batch, "frames_per_call": frames_per_call, "half": a.half,
        "mean_ms_per_frame_predict": round(1000 * t_pred / nf, 3),
        "mean_ms_per_frame_e2e": round(1000 * t_all / nf, 3),
    }
    backends = [det] if a.mode == "full" else [x for x in (det.backend, det.full_backend) if x is not None]
    timing["ultralytics_speed_ms_per_frame"] = {
        k: round(sum(bk.speed_ms[k] for bk in backends) / nf, 3) for k in ("preprocess", "inference", "postprocess")}
    if a.mode == "tiled":
        timing["tiled_breakdown_ms_per_frame"] = {
            "tile_predict": round(1000 * det.timing["tile_infer_s"] / nf, 3),
            "full_frame_predict": round(1000 * det.timing["full_infer_s"] / nf, 3),
            "merge": round(1000 * det.timing["merge_s"] / nf, 3),
            "tiles_per_frame": round(det.timing["tiles"] / nf, 3)}
    finished = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    common = {
        "detector": a.name,
        "weights": str(a.weights.resolve()), "weights_sha256": sha256(a.weights),
        "mode": a.mode,
        "classes_map": {str(c): b for c, b in bench_of.items()},
        "save_conf": a.conf, "v1_conf": a.v1_conf,
        "imgsz_arg": a.imgsz, "imgsz_used": imgsz_used,
        "end2end_arg": a.end2end, "head_end2end_used": head_e2e,
        "max_det": a.max_det,
        "predict_kwargs": desc.get("predict_kwargs") or desc["tile_backend"]["predict_kwargs"],
        "tiling": {k: v for k, v in desc.items() if k not in ("tile_backend", "backend")} if a.mode == "tiled" else None,
        "ultralytics": ul_version, "torch": torch.__version__ if torch else None,
        "cuda": torch.version.cuda if torch else None,
        "gpu": None if ort_cpu or not device.startswith("cuda") else (
            torch.cuda.get_device_name(0) if torch else _gpu_name()),
        "frame_index": "1-based position in sorted img1/*<imExt>, == seqLength",
        "cache_format": "<root>/<detector>/<bench>/<seq>.npy, (N,6) [frame,x1,y1,x2,y2,conf]",
    }
    if a.backend != "torch":  # absent key = torch (caches written before backends existed)
        common["backend"] = a.backend
        common["backend_detail"] = {k: v for k, v in desc["tile_backend"].items() if k != "predict_kwargs"}
    if device == "cpu" or ort_cpu:
        common["cpu"] = _cpu_model()
        common["threads"] = a.threads
    if len(bench_of) == 1 and not a.classes:
        common["class_kept"] = a.cls  # legacy key
    meta_path = a.out_root / a.name / "meta.json"
    with locked(a.out_root / a.name / ".meta.lock"):
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        guard = ("weights_sha256", "save_conf", "imgsz_used", "head_end2end_used", "mode", "tiling", "classes_map")
        defaults = {"mode": "full"}  # caches written before modes existed were full-frame
        clash = {k: (meta.get(k, defaults.get(k)), common[k]) for k in guard
                 if meta and (k in meta or k in defaults) and meta.get(k, defaults.get(k)) != common[k]}
        if clash:
            raise SystemExit(f"meta.json disagrees with this run (old, new): {clash}")
        meta.update(common)
        for b in bench_dirs:
            bs = bench_seqs[b]
            meta.setdefault("runs", {})[f"{b}/{a.split or 'seqs'}"] = {
                "bench": b, "bench_dir": str(bench_dirs[b]), "split": a.split, "seqs": bs,
                "frames": sum(per_seq[s]["frames"] for s in bs),
                "job_seqs": seqs, "job_timing": timing,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "finished": finished,
                "per_seq": {s: {k: v for k, v in per_seq[s].items() if k not in bench_dirs or k == b}
                            for s in bs}}
        cache.write_meta(a.out_root, a.name, meta)
    print(f"[detect] done: {tot_frames} frames, predict {timing['mean_ms_per_frame_predict']} ms/f, "
          f"e2e {timing['mean_ms_per_frame_e2e']} ms/f, meta -> {meta_path}", flush=True)

if __name__ == "__main__":
    main()
