"""Pre-render the Hugging Face Space's example clips from VISO (run on Polaris, GPU).

    python scripts/space_examples.py --out space/examples [--device cuda:0]

VISO (CC BY-NC-SA 4.0) was never used for training or tuning, so every clip is
cross-dataset. For each VISO sequence:
  * the protected zone is a circle (r = 25 px, the threat evaluation's radius) placed
    on the path of the sequence's longest-moving labelled object, 8 s (the evaluation's
    lead time) after the first anchor where it moves. It is placed from the labels on
    purpose, to show the warning; the page says so.
  * the full pipeline runs (tiled YOLO26s, per-class TALOS-Track, zone-entry threat).
  * the annotated video is re-encoded to H.264 for browsers.
  * honest per-clip numbers go to summary.json: when the labelled object really enters
    the zone (from the labels), which predicted track follows it, when that track's first
    alert came (alerts on other tracks do not count), the lead time, a one-line verdict
    (warned / late / missed) and detector recall on the labelled (moving) objects at IoU >= 0.3.
    Clips are listed best first; failures are kept, not dropped.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from src.TALOS.data.mot import read_mot, read_seqinfo
from src.TALOS.pipeline.run import CONFIG, Pipeline, run_video
from src.TALOS.threat.model import Zone

RADIUS, LEAD_S = 25.0, 8.0
BENCHES = {"VISO_plane-all": "plane", "VISO_ship-all": "ship"}


def pick_zone(gt: np.ndarray, fps: float):
    """(zone, track id, GT entry frame) on the longest-moving labelled track, or None."""
    lead = int(round(LEAD_S * fps))
    best = None
    for tid in np.unique(gt[:, 1]):
        t = gt[gt[:, 1] == tid]
        t = t[np.argsort(t[:, 0])]
        c = np.c_[t[:, 2] + t[:, 4] / 2, t[:, 3] + t[:, 5] / 2]
        if len(t) <= lead + 2:
            continue
        travel = np.linalg.norm(c[-1] - c[0])
        if best is None or travel > best[0]:
            best = (travel, tid, t, c)
    if best is None or best[0] < 2 * RADIUS:
        return None
    _, tid, t, c = best
    a = 0
    for a in range(len(c) - lead):  # first anchor where it moves at least a zone radius in the lead time
        if np.linalg.norm(c[a + lead] - c[a]) >= RADIUS:
            break
    z = Zone.circle(*c[a + lead], RADIUS)
    inside = [int(f) for f, (x, y) in zip(t[:, 0], c) if np.hypot(x - c[a + lead][0], y - c[a + lead][1]) <= RADIUS]
    return z, int(tid), (min(inside) if inside else None), c[a + lead].tolist()


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def recall(gt: np.ndarray, tracks_csv: Path, cls: str, thr: float = 0.3) -> float:
    rows = [r.split(",") for r in tracks_csv.read_text().splitlines()[1:]]
    pred = {}
    for r in rows:
        if r[1] == cls:
            pred.setdefault(int(r[0]), []).append([float(v) for v in r[3:7]])
    hit = 0
    for f in np.unique(gt[:, 0]):
        g = gt[gt[:, 0] == f]
        gb = np.c_[g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]]
        p = np.array(pred.get(int(f), []), float).reshape(-1, 4)
        if len(p):
            hit += int((iou(gb, p).max(axis=1) >= thr).sum())
    return round(hit / max(len(gt), 1), 3)


def matched_track(g: np.ndarray, tracks_csv: Path, thr: float = 0.3):
    """The predicted (class, id) that overlaps labelled track ``g`` (IoU >= thr) on the most frames."""
    count = {}
    for r in tracks_csv.read_text().splitlines()[1:]:
        x = r.split(",")
        gg = g[g[:, 0] == int(x[0])]
        if len(gg):
            gb = np.c_[gg[:, 2], gg[:, 3], gg[:, 2] + gg[:, 4], gg[:, 3] + gg[:, 5]]
            if iou(gb, np.array([[float(v) for v in x[3:7]]]))[0, 0] >= thr:
                count[(x[1], int(x[2]))] = count.get((x[1], int(x[2])), 0) + 1
    return max(count, key=count.get) if count else None


def verdict(r: dict) -> str:
    if r["zone"] is None:
        return "no zone (no labelled object moves far enough)"
    if r["matched_track"] is None:
        return "missed: the detector never found the labelled object"
    if r["lead_s"] is None:
        return "missed: tracked, but no alert before it entered the zone"
    if r["lead_s"] < 0:
        return f"late: first alert {-r['lead_s']} s after it entered the zone"
    return f"warned {r['lead_s']} s before it entered the zone"


def main() -> None:
    from src.TALOS.detect.tiled import TiledDetector

    ap = argparse.ArgumentParser()
    ap.add_argument("--mot-root", type=Path, default=Path("~/talos/data/mot").expanduser())
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--work", type=Path, default=Path("~/talos/runs/space").expanduser())
    a = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    d = cfg["detector"]
    det = TiledDetector(str(Path(d["weights"]).expanduser()), tile=d["tile"], overlap=d["overlap"],
                        imgsz=d["imgsz"], conf=d["conf"], device=a.device)
    ffmpeg = shutil.which("ffmpeg")
    summary = []
    for bench, cls in BENCHES.items():
        for seq in sorted(p for p in (a.mot_root / bench).iterdir() if (p / "seqinfo.ini").exists()):
            info = read_seqinfo(seq)
            gt = read_mot(seq / "gt" / "gt.txt")
            picked = pick_zone(gt, info.fps)
            zone, tid, entry, centre = picked if picked else (None, None, None, None)
            work = a.work / seq.name
            pipe = Pipeline(det, fps=info.fps, zone=zone, config=cfg, threat_every=1)
            timing = run_video(seq / "img1", work, pipe, batch=4)
            alerts = [json.loads(x) for x in (work / "alerts.jsonl").read_text().splitlines()]
            g = gt[gt[:, 1] == tid] if tid is not None else gt[:0]
            match = matched_track(g, work / "tracks.csv") if len(g) else None
            # lead time counts only alerts on the track that follows the labelled object
            first = min((x["frame"] for x in alerts if match and (x["class"], x["id"]) == match), default=None)
            others = sorted({(x["class"], x["id"]) for x in alerts} - ({match} if match else set()))
            dst = a.out / seq.name
            dst.mkdir(parents=True, exist_ok=True)
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(work / "annotated.mp4"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow", "-movflags", "+faststart",
                            str(dst / "annotated.mp4")], check=True)
            shutil.copy(work / "alerts.jsonl", dst / "alerts.jsonl")
            shutil.copy(work / "tracks.csv", dst / "tracks.csv")
            row = {"seq": seq.name, "class": cls, "size": [info.width, info.height], "frames": info.length,
                   "fps": info.fps, "gt_tracks": int(len(np.unique(gt[:, 1]))),
                   "zone": None if zone is None else {"centre": [round(v, 1) for v in centre], "radius": RADIUS,
                                                       "placed_on_gt_track": tid},
                   "matched_track": None if match is None else list(match),
                   "gt_entry_frame": entry, "first_alert_frame": first,
                   "other_alerting_tracks": [list(o) for o in others],
                   "lead_s": None if (entry is None or first is None) else round((entry - first) / info.fps, 1),
                   "alerts": len(alerts), "recall_iou0.3": recall(gt, work / "tracks.csv", cls),
                   "a100_ms_per_frame": timing["ms_per_frame"]["total"]}
            row["verdict"] = verdict(row)
            summary.append(row)
            print(json.dumps(row), flush=True)
    # best first: warned (longest lead), then late, then missed
    summary.sort(key=lambda r: (r["lead_s"] is None, -(r["lead_s"] or 0), -r["recall_iou0.3"]))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
