"""TALOS on AWS Lambda: POST a short satellite clip, get tracks and zone-entry alerts back.

Same pipeline as ``python -m src.TALOS.pipeline.run`` (tiled YOLO26 on ONNX Runtime CPU,
TALOS-Track, zone-entry threat), wrapped for a Lambda function URL.

    GET  /            usage and limits
    POST /            JSON body:
        video_b64     base64 of an mp4/avi clip (or image_b64 for one frame)
        zone_circle   [cx, cy, r] in pixels of the uploaded clip (optional)
        zone          [[x, y], ...] polygon instead of a circle (optional)
        fps           frame rate of the clip (default 10)
        max_frames    frames to process (default and cap: MAX_FRAMES)
        return_frame  true: also return the last annotated frame as JPEG base64

Response: frames, tracks (one row per track: class, id, first/last frame, max p_entry),
alerts (one per frame and track over the val-tuned alert threshold), timing.

Limits keep one call inside the free tier: Lambda's 6 MB request body, MAX_FRAMES frames,
and frames downscaled so the long side is at most MAX_SIDE (the Hugging Face demo does the same).
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from src.TALOS.detect.tiled import TiledDetector
from src.TALOS.pipeline.run import CONFIG, Pipeline
from src.TALOS.threat.model import Zone

MODEL = Path(os.environ.get("TALOS_MODEL", "/opt/talos/talos_y26s_t1024.onnx"))
MAX_FRAMES = int(os.environ.get("TALOS_MAX_FRAMES", "40"))
MAX_SIDE = int(os.environ.get("TALOS_MAX_SIDE", "1280"))
CFG = json.loads(CONFIG.read_text())

_detector = None


def cpu_limit() -> int:
    """CPUs this process may use (os.cpu_count() reports the host; oversubscribing ORT is ~40x slower)."""
    if os.environ.get("TALOS_THREADS"):
        return max(1, int(os.environ["TALOS_THREADS"]))
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, -(-int(quota) // int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def detector():
    """Built once per container, so warm calls skip the model load."""
    global _detector
    if _detector is None:
        d = CFG["detector"]
        _detector = TiledDetector(str(MODEL), tile=d["tile"], overlap=d["overlap"], imgsz=d["imgsz"],
                                  conf=d["conf"], device="cpu", backend="onnx", ort_provider="cpu",
                                  threads=cpu_limit(), batch=8)
    return _detector


class BadRequest(ValueError):
    pass


def _reply(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}


def _parse(event: dict) -> tuple[str, dict]:
    """Function URL events carry an HTTP body; a direct ``lambda invoke`` passes the dict itself."""
    if "requestContext" not in event:
        return "POST", event
    method = event["requestContext"].get("http", {}).get("method", "POST")
    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode()
    if method != "POST":
        return method, {}
    try:
        return method, json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        raise BadRequest(f"body is not JSON: {e}") from None


def _frames(req: dict, limit: int) -> list[np.ndarray]:
    if "image_b64" in req:
        img = cv2.imdecode(np.frombuffer(base64.b64decode(req["image_b64"]), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise BadRequest("image_b64 is not a decodable image")
        return [img]
    if "video_b64" not in req:
        raise BadRequest("send video_b64 (a short clip) or image_b64")
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir="/tmp") as f:
        f.write(base64.b64decode(req["video_b64"]))
        f.flush()
        cap = cv2.VideoCapture(f.name)
        frames = []
        while len(frames) < limit:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
    if not frames:
        raise BadRequest("video_b64 has no decodable frames")
    return frames


def _zone(req: dict, scale: float) -> Zone | None:
    if req.get("zone_circle"):
        cx, cy, r = (float(v) * scale for v in req["zone_circle"])
        return Zone.circle(cx, cy, r)
    if req.get("zone"):
        pts = np.asarray(req["zone"], float).reshape(-1, 2) * scale
        if len(pts) < 3:
            raise BadRequest("zone needs at least 3 [x, y] points")
        return Zone(pts)
    return None


def process(req: dict, det=None) -> dict:
    t0 = time.perf_counter()
    limit = min(int(req.get("max_frames") or MAX_FRAMES), MAX_FRAMES)
    frames = _frames(req, limit)
    h, w = frames[0].shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        size = (round(w * scale), round(h * scale))
        frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
    fps = float(req.get("fps") or 10.0)
    pipe = Pipeline(det or detector(), fps=fps, zone=_zone(req, scale), config=CFG)

    tracks: dict[tuple[str, int], dict] = {}
    alerts, rows = [], []
    for s in range(0, len(frames), 8):
        chunk = frames[s:s + 8]
        t1 = time.perf_counter()
        dets = pipe.detector.predict(chunk)
        pipe.timer.add("detect", time.perf_counter() - t1)
        for fr, d in zip(chunk, dets):
            rows, al = pipe.step(fr, np.asarray(d, float).reshape(-1, 6))
            alerts.extend(al)
            for r in rows:
                t = tracks.setdefault((r["class"], r["id"]), {"class": r["class"], "id": r["id"],
                                                              "first_frame": r["frame"], "max_p_entry": 0.0})
                t["last_frame"] = r["frame"]
                t["max_p_entry"] = max(t["max_p_entry"], r["p_entry"])
    out ={"frames": len(frames), "input_size": [w, h], "processed_scale": round(scale, 4), "fps": fps,
           "alert_threshold": pipe.alert_thr, "tracks": sorted(tracks.values(), key=lambda t: (t["class"], t["id"])),
           "alerts": alerts,
           "timing": {"ms_per_frame": pipe.timer.per_frame_ms(len(frames)),
                      "wall_s": round(time.perf_counter() - t0, 2)}}
    if req.get("return_frame"):
        ok, jpg = cv2.imencode(".jpg", pipe.render(frames[-1], rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
        out["last_frame_jpeg_b64"] = base64.b64encode(jpg.tobytes()).decode() if ok else None
    return out


USAGE = {
    "service": "TALOS: aircraft and ship tracking with zone-entry alerts in satellite video",
    "post": {"video_b64": "base64 mp4 clip (<= ~4 MB before base64)", "zone_circle": "[cx, cy, r] px",
             "zone": "[[x, y], ...] px", "fps": "default 10", "max_frames": f"cap {MAX_FRAMES}",
             "return_frame": "bool"},
    "limits": {"max_frames": MAX_FRAMES, "max_side_px": MAX_SIDE, "request_body": "6 MB (Lambda)"},
    "code": "https://github.com/Gakshith/Tactical-Aerospace-Localization-Objective-Scoring-TALOS-",
}


def handler(event, context=None):
    try:
        method, req = _parse(event or {})
        if method == "GET":
            return _reply(200, USAGE)
        if method != "POST":
            return _reply(405, {"error": f"{method} not allowed; use GET or POST"})
        return _reply(200, process(req))
    except BadRequest as e:
        return _reply(400, {"error": str(e)})
    except Exception as e:  # noqa: BLE001 - one bad clip must not look like a platform error
        print(f"ERROR {type(e).__name__}: {e}")
        return _reply(500, {"error": f"{type(e).__name__}: {e}"})
