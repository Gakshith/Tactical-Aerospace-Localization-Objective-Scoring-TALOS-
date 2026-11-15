"""TALOS v2 end to end: video in, tracked and threat-ranked video out.

    python -m src.TALOS.pipeline.run VIDEO --out OUT_DIR --zone-circle CX,CY,R
    python -m src.TALOS.pipeline.run VIDEO --out OUT_DIR --zone X1,Y1,X2,Y2,X3,Y3,...

Stages, each timed separately:
  decode   read frames (OpenCV)
  detect   tiled YOLO26 (TiledDetector; any backend it supports)
  track    one TALOS-Track per class, with the tuned settings in configs/talos_v2.json
  threat   P(enter zone within H s) for each confirmed track, every ``threat_every`` frames
  render   boxes, IDs and risk drawn onto the frame, written to an mp4

Outputs in OUT_DIR:
  annotated.mp4   the video with overlays (red = alerting, amber = rising, green = clear)
  tracks.csv      frame, class, id, x1, y1, x2, y2, conf
  alerts.jsonl    one line per (frame, track) whose entry probability crosses the
                  alert threshold, with p_entry and median time-to-entry in seconds
  timing.json     per-stage milliseconds per frame, frames, hardware

The alert threshold is the 5%-false-alarm operating point measured on AIR-MOT
val, not a hand-picked number. Distances are pixels; times use ``--fps``
(AIR-MOT does not state its frame rate; 10 fps is assumed and is the default).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.TALOS.threat.model import ThreatConfig, Zone, entry_probability
from src.TALOS.track.talos_track import CONFIRMED, LOST, Config, TalosTracker

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "talos_v2.json"
COLORS = {"alert": (40, 40, 230), "rising": (0, 170, 255), "clear": (90, 200, 90)}


@dataclass
class StageTimer:
    totals: dict = field(default_factory=lambda: {k: 0.0 for k in
                                                  ("decode", "detect", "track", "threat", "render")})

    def add(self, stage: str, seconds: float) -> None:
        self.totals[stage] += seconds

    def per_frame_ms(self, frames: int) -> dict:
        out = {k: round(1000 * v / max(frames, 1), 3) for k, v in self.totals.items()}
        out["total"] = round(sum(out.values()), 3)
        return out


class Pipeline:
    def __init__(self, detector, fps: float = 10.0, zone: Zone | None = None,
                 config: dict | None = None, threat_every: int = 5):
        self.cfg = config or json.loads(CONFIG.read_text())
        self.detector = detector
        self.fps = fps
        self.zone = zone
        self.class_names = {int(k): v for k, v in self.cfg["detector"]["classes"].items()}
        self.trackers = {name: TalosTracker(fps, Config(**self.cfg["tracker"][name]["params"]))
                         for name in self.class_names.values()}
        th = self.cfg["threat"]
        self.threat_cfg = ThreatConfig(horizon_s=th["horizon_s"], fps=fps, accel_std=th["accel_std"])
        self.alert_thr = float(th["alert_threshold"])
        self.threat_every = max(1, threat_every)
        self.risk: dict[tuple[str, int], tuple[float, float]] = {}   # (class, id) -> (p, tte)
        self.timer = StageTimer()
        self.frame_no = 0

    def _threat(self, name: str, trk: TalosTracker) -> None:
        if self.zone is None:
            return
        reach_speed_cap = 3.0  # skip tracks that cannot plausibly reach the zone in time
        cx, cy = self.zone.centre
        radius = float(np.max(np.linalg.norm(self.zone.polygon - self.zone.centre, axis=1)))
        for t in trk.tracks:
            if t.state not in (CONFIRMED, LOST):
                continue
            pos, vel = t.x[:2], t.x[2:4]
            sigma_v = float(np.sqrt(max(t.P[2, 2], t.P[3, 3])))
            reach = (np.linalg.norm(vel) + reach_speed_cap * sigma_v) * self.threat_cfg.horizon_s
            if np.hypot(pos[0] - cx, pos[1] - cy) - radius > reach:
                self.risk[(name, t.id)] = (0.0, float("inf"))
                continue
            self.risk[(name, t.id)] = entry_probability(t.x, t.P, self.zone, self.threat_cfg)

    def step(self, frame: np.ndarray, dets: np.ndarray) -> tuple[list[dict], list[dict]]:
        """Track and score one frame's detections; returns (track rows, alerts)."""
        self.frame_no += 1
        rows, alerts = [], []
        t0 = time.perf_counter()
        outs = {}
        for cid, name in self.class_names.items():
            d = dets[dets[:, 5] == cid][:, :5] if len(dets) else np.zeros((0, 5))
            outs[name] = self.trackers[name].update(d)
        t1 = time.perf_counter()
        if self.zone is not None and self.frame_no % self.threat_every == 1 % self.threat_every:
            for name, trk in self.trackers.items():
                self._threat(name, trk)
        t2 = time.perf_counter()
        self.timer.add("track", t1 - t0)
        self.timer.add("threat", t2 - t1)
        for name, out in outs.items():
            for x1, y1, x2, y2, tid, conf in out:
                tid = int(tid)
                row = {"frame": self.frame_no, "class": name, "id": tid, "x1": round(x1, 1),
                       "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1), "conf": round(conf, 3)}
                p, tte = self.risk.get((name, tid), (0.0, float("inf")))
                row["p_entry"] = round(p, 3)
                rows.append(row)
                if p >= self.alert_thr:
                    alerts.append({"frame": self.frame_no, "time_s": round(self.frame_no / self.fps, 2),
                                   "class": name, "id": tid, "p_entry": round(p, 3),
                                   "time_to_entry_s": None if not np.isfinite(tte) else round(tte, 2)})
        return rows, alerts

    def render(self, frame: np.ndarray, rows: list[dict]) -> np.ndarray:
        t0 = time.perf_counter()
        img = frame.copy()
        if self.zone is not None:
            cv2.polylines(img, [self.zone.polygon.astype(np.int32)], True, (255, 255, 255), 2)
        for r in sorted(rows, key=lambda r: r["p_entry"]):
            p = r["p_entry"]
            key = "alert" if p >= self.alert_thr else ("rising" if p >= self.alert_thr / 3 else "clear")
            c = COLORS[key]
            p1, p2 = (int(r["x1"]), int(r["y1"])), (int(r["x2"]), int(r["y2"]))
            cv2.rectangle(img, p1, p2, c, 2)
            label = f"{r['class'][0].upper()}{r['id']}" + (f" {p:.0%}" if p >= self.alert_thr / 3 else "")
            cv2.putText(img, label, (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        self.timer.add("render", time.perf_counter() - t0)
        return img


class FrameFolder:
    """A folder of frames (e.g. an AIR-MOT img1/ dir) read like a VideoCapture."""
    def __init__(self, path: Path):
        exts = (".jpg", ".jpeg", ".png")
        self.files = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in exts)
        self.i = 0

    def isOpened(self) -> bool:
        return bool(self.files)

    def read(self):
        if self.i >= len(self.files):
            return False, None
        img = cv2.imread(str(self.files[self.i]))
        self.i += 1
        return img is not None, img

    def release(self) -> None:
        pass


def run_video(video: Path, out: Path, pipe: Pipeline, batch: int = 4, max_frames: int | None = None,
              write_video: bool = True) -> dict:
    """``video`` is a video file or a folder of frames."""
    out.mkdir(parents=True, exist_ok=True)
    cap = FrameFolder(video) if Path(video).is_dir() else cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    writer = None
    n = 0
    with (out / "tracks.csv").open("w") as ftr, (out / "alerts.jsonl").open("w") as fal:
        ftr.write("frame,class,id,x1,y1,x2,y2,conf,p_entry\n")
        done = False
        while not done:
            t0 = time.perf_counter()
            frames = []
            while len(frames) < batch:
                ok, fr = cap.read()
                if not ok or (max_frames and n + len(frames) >= max_frames):
                    done = True
                    break
                frames.append(fr)
            pipe.timer.add("decode", time.perf_counter() - t0)
            if not frames:
                break
            t1 = time.perf_counter()
            dets = pipe.detector.predict(frames)
            pipe.timer.add("detect", time.perf_counter() - t1)
            for fr, d in zip(frames, dets):
                rows, alerts = pipe.step(fr, np.asarray(d, float).reshape(-1, 6))
                for r in rows:
                    ftr.write(",".join(str(r[k]) for k in
                                       ("frame", "class", "id", "x1", "y1", "x2", "y2", "conf", "p_entry")) + "\n")
                for a in alerts:
                    fal.write(json.dumps(a) + "\n")
                if write_video:
                    img = pipe.render(fr, rows)
                    if writer is None:
                        writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 pipe.fps, (img.shape[1], img.shape[0]))
                    writer.write(img)
            n += len(frames)
    cap.release()
    if writer is not None:
        writer.release()
    timing = {"frames": n, "ms_per_frame": pipe.timer.per_frame_ms(n),
              "fps_end_to_end": round(1000 / max(pipe.timer.per_frame_ms(n)["total"], 1e-9), 2),
              "host": platform.node(), "machine": platform.machine(), "processor": platform.processor(),
              "detector": getattr(pipe.detector, "describe", lambda: type(pipe.detector).__name__)()}
    (out / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    return timing


def parse_zone(a) -> Zone | None:
    if a.zone_circle:
        cx, cy, r = (float(v) for v in a.zone_circle.split(","))
        return Zone.circle(cx, cy, r)
    if a.zone:
        v = [float(x) for x in a.zone.split(",")]
        if len(v) < 6 or len(v) % 2:
            raise SystemExit("--zone needs at least 3 x,y vertices")
        return Zone(np.array(v).reshape(-1, 2))
    return None


def main() -> None:
    from src.TALOS.detect.tiled import TiledDetector

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video", type=Path, help="video file or folder of frames")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", type=str, default=None, help="default: configs/talos_v2.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--zone-circle", help="cx,cy,r in pixels")
    ap.add_argument("--zone", help="polygon x1,y1,x2,y2,... in pixels")
    ap.add_argument("--batch", type=int, default=4, help="frames per detector call")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    d = cfg["detector"]
    det = TiledDetector(str(Path(a.weights or d["weights"]).expanduser()), tile=d["tile"], overlap=d["overlap"],
                        imgsz=d["imgsz"], conf=d["conf"], device=a.device)
    pipe = Pipeline(det, fps=a.fps, zone=parse_zone(a), config=cfg)
    timing = run_video(a.video, a.out, pipe, batch=a.batch, max_frames=a.max_frames,
                       write_video=not a.no_video)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
