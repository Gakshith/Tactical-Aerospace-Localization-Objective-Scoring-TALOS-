"""TALOS on Hugging Face Spaces: aircraft and ship tracking with zone-entry warnings
in satellite video. CPU only (ONNX Runtime); every number shown comes from a file in
results/ that the repo's Slurm jobs wrote.

Layout of this folder (assembled by scripts/build_space.sh):
  src/TALOS/...            the same pipeline code as the repo (no torch, no ultralytics)
  configs/talos_v2.json    tuned tracker settings and the alert threshold
  models/talos_y26s_t1024.onnx   YOLO26s fine-tuned on AIR-MOT tiles, end2end head
  examples/<viso-seq>/     pre-rendered VISO clips + summary.json
  results/*.json           the measured tables
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from src.TALOS.detect.tiled import TiledDetector
from src.TALOS.pipeline.run import CONFIG, Pipeline, run_video
from src.TALOS.threat.model import Zone

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE / "examples"
RESULTS = HERE / "results"
MODEL = HERE / "models" / "talos_y26s_t1024.onnx"
MAX_FRAMES = 40          # free CPU: keep a run under about a minute
MAX_SIDE = 1280          # longer sides are downscaled (2 tiles of 1024 instead of 6)
THREADS = os.cpu_count() or 2
CFG = json.loads(CONFIG.read_text())

_detector = None


def detector() -> TiledDetector:
    global _detector
    if _detector is None:
        d = CFG["detector"]
        _detector = TiledDetector(str(MODEL), tile=d["tile"], overlap=d["overlap"], imgsz=d["imgsz"],
                                  conf=d["conf"], device="cpu", backend="onnx", ort_provider="cpu",
                                  threads=THREADS, batch=8)
    return _detector


def h264(src: Path, dst: Path) -> Path:
    ff = shutil.which("ffmpeg")
    if ff is None:
        return src
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "26", "-movflags", "+faststart", str(dst)], check=True)
    return dst


# ---------------------------------------------------------------- examples tab
def example_rows() -> list[dict]:
    p = EXAMPLES / "summary.json"
    return json.loads(p.read_text()) if p.exists() else []


def example_label(r: dict) -> str:
    return f"{r['seq']} · {r['class']} · {r['verdict']}"


def show_example(label: str):
    r = next(x for x in example_rows() if example_label(x) == label)
    d = EXAMPLES / r["seq"]
    alerts = [json.loads(x) for x in (d / "alerts.jsonl").read_text().splitlines()]
    first = {}
    for a in alerts:
        first.setdefault((a["class"], a["id"]), a)
    table = [[f"{k[0]} {k[1]}", a["frame"], a["time_s"], a["p_entry"], a["time_to_entry_s"]]
             for k, a in first.items()]
    lines = [f"**Zone:** circle of {r['zone']['radius']:.0f} px placed on the path of a labelled "
             f"{r['class']} (track {r['zone']['placed_on_gt_track']}), from the labels, to show the warning."
             if r["zone"] else "**Zone:** none (no labelled object moves far enough).",
             f"**Detector recall on labelled objects** (IoU ≥ 0.3, after tracking): {r['recall_iou0.3']:.0%}. "
             "VISO labels only moving objects, so parked aircraft appear as extra boxes.",
             f"**Alerts:** {r['alerts']} track-frames over the threshold."]
    lines.insert(0, f"### Result: {r['verdict']}")
    if r["matched_track"] and r["gt_entry_frame"]:
        lines.append(f"The labelled object (followed by track {r['matched_track'][0]} {r['matched_track'][1]}) "
                     f"enters the zone at frame {r['gt_entry_frame']}"
                     + (f"; that track's first alert came at frame {r['first_alert_frame']}." if r["first_alert_frame"]
                        else "."))
    if r["other_alerting_tracks"]:
        lines.append("Other tracks that alerted: " + ", ".join(f"{c} {i}" for c, i in r["other_alerting_tracks"]))
    return str(d / "annotated.mp4"), "\n\n".join(lines), table


# ---------------------------------------------------------------- your clip tab
def first_frame(video):
    if not video:
        return None
    cap = cv2.VideoCapture(video)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def scaled(fr):
    h, w = fr.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    return fr if s == 1.0 else cv2.resize(fr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def preview(video, zx, zy, zr):
    fr = first_frame(video)
    if fr is None:
        return None
    fr = scaled(fr)
    h, w = fr.shape[:2]
    cv2.circle(fr, (int(zx / 100 * w), int(zy / 100 * h)), int(zr), (255, 255, 255), 2)
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def run_clip(video, zx, zy, zr, progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload a clip first.")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    tmp = Path(tempfile.mkdtemp())
    frames_dir = tmp / "img1"
    frames_dir.mkdir()
    n = 0
    while n < MAX_FRAMES:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        cv2.imwrite(str(frames_dir / f"{n:06d}.png"), scaled(fr))
    cap.release()
    if n == 0:
        raise gr.Error("Could not read any frames from that file.")
    h, w = cv2.imread(str(frames_dir / "000001.png")).shape[:2]
    zone = Zone.circle(zx / 100 * w, zy / 100 * h, zr)
    progress(0.1, desc=f"Detecting and tracking {n} frames on CPU")
    t = time.time()
    pipe = Pipeline(detector(), fps=fps, zone=zone, config=CFG, threat_every=1)
    timing = run_video(frames_dir, tmp / "out", pipe, batch=2)
    out = h264(tmp / "out" / "annotated.mp4", tmp / "out" / "annotated_h264.mp4")
    alerts = [json.loads(x) for x in (tmp / "out" / "alerts.jsonl").read_text().splitlines()]
    tracks = (tmp / "out" / "tracks.csv").read_text().splitlines()[1:]
    ids = {(r.split(",")[1], r.split(",")[2]) for r in tracks}
    ms = timing["ms_per_frame"]
    note = (f"{n} frames at {w}×{h}, {fps:.0f} fps in {time.time() - t:.0f} s on {THREADS} CPU threads. "
            f"Per frame: detect {ms['detect']:.0f} ms, track {ms['track']:.1f} ms, threat {ms['threat']:.1f} ms. "
            f"{len(ids)} tracks, {len(alerts)} alert track-frames.")
    table = [[f"{a['class']} {a['id']}", a["frame"], a["time_s"], a["p_entry"], a["time_to_entry_s"]] for a in alerts]
    return str(out), note, table


# ---------------------------------------------------------------- results tab
def results_md() -> str:
    out = []
    names = {"v1": "TALOS v1", "sort": "SORT", "bytetrack": "ByteTrack", "ocsort": "OC-SORT",
             "botsort": "BoT-SORT", "talos": "**TALOS-Track**"}
    for cls in ("plane", "ship"):
        p = RESULTS / f"tracking-test-{cls}.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())["scores"]
        out.append(f"#### Tracking, AIR-MOT test, {cls}s (every tracker tuned on val with 30 configs, same detections)\n")
        out.append("| tracker | HOTA | IDF1 | MOTA | ID switches |\n|---|---|---|---|---|")
        for k, label in names.items():
            if k in s:
                r = s[k]
                out.append(f"| {label} | {r['HOTA']:.1f} | {r['IDF1']:.1f} | {r['MOTA']:.1f} | {int(r['IDSW'])} |")
        out.append("")
    rows = []
    for name, f in (("AIR-MOT val", "threat-airmot-val.json"), ("VISO (never trained on)", "threat-viso.json")):
        p = RESULTS / f
        if p.exists():
            d = json.loads(p.read_text())
            v2, v1 = d["v2_entry_prob"], d["v1_score"]
            rows.append(f"| {name} | {v2['auroc']:.3f} | {v1['auroc']:.3f} | {v2['avg_precision']:.3f} | "
                        f"{v1['avg_precision']:.3f} | {v2['entries_warned']} | {v2['median_lead_s']} s |")
    if rows:
        out.append("#### Threat: does P(enter zone within 10 s) rank zones that really get entered?\n")
        out.append("| data | AUROC v2 | AUROC v1 | AP v2 | AP v1 | entries warned (v2, 5% false alarms) | median lead |")
        out.append("|---|---|---|---|---|---|---|")
        out += rows + [""]
    p = RESULTS / "detector_speed.json"
    if p.exists():
        d = json.loads(p.read_text())
        out.append("#### Detector runtimes, one A100, AIR-MOT val (1920×1080, 6 tiles per frame)\n")
        out.append("| runtime | ms / frame | speed-up | plane AP50 | ship AP50 | matches PyTorch |\n|---|---|---|---|---|---|")
        for r in sorted(d["rows"], key=lambda r: r["detect_ms_per_frame"]):
            c = r["classes"]
            out.append(f"| {r['variant'].removeprefix('rt_')} | {r['detect_ms_per_frame']:.1f} | "
                       f"{r['speedup_vs_ref']:.2f}× | {c['plane']['ap50']:.3f} | {c['ship']['ap50']:.3f} | "
                       f"{'yes' if r['parity_ok'] else 'no'} |")
        out.append("\nMatch = AP50 within 0.005 and box count within 1% of PyTorch fp32.")
    return "\n".join(out) or "Results files are missing from this build."


ABOUT = """
**TALOS** detects, tracks and threat-ranks aircraft and ships in satellite video.
- **Detect:** YOLO26s on 1024 px tiles with an edge-aware merge. It was fine-tuned on AIR-MOT (Jilin-1 video), and this page runs it as ONNX on CPU.
- **Track:** one Kalman tracker per class, with Mahalanobis gating, two-stage association and duplicate suppression.
- **Threat:** the probability that each track enters the zone within 10 s, Monte Carlo sampled from the filter's state and covariance. The alert threshold is the 5%-false-alarm point measured on AIR-MOT val.

**Limits:**
- Aircraft under 16 px are mostly missed (val recall 5%).
- Ships have many false positives.
- Distances are in pixels, and times assume the clip's frame rate (10 fps for AIR-MOT, which does not state one).

**Data and license.**
- The example clips are from **VISO**, licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The source is Q. Yin et al., *Detecting and Tracking Small and Dense Moving Objects in Satellite Videos: A Benchmark*, IEEE TGRS 2021 ([github.com/QingyongHu/VISO](https://github.com/QingyongHu/VISO)). The annotated clips are shared under the same license, for non-commercial use.
- The detector was trained on **AIR-MOT**: Q. He et al., *Multi-Object Tracking in Satellite Videos with Graph-Based Multi-Task Modeling*, IEEE TGRS 2022 ([github.com/HeQibin/TGraM](https://github.com/HeQibin/TGraM)). No AIR-MOT frames are shown here.
- The code is MIT-licensed.
"""


def build() -> gr.Blocks:
    rows = example_rows()
    labels = [example_label(r) for r in rows]
    with gr.Blocks(title="TALOS: satellite aircraft tracking and zone warnings") as demo:
        gr.Markdown("# TALOS\nAircraft and ship tracking with zone-entry warnings in satellite video. "
                    "Boxes are **red** when the track is likely to enter the white zone within 10 s, "
                    "**amber** when rising, **green** when clear.")
        with gr.Tab("Satellite examples"):
            n_ok = sum(1 for r in rows if r["lead_s"] is not None and r["lead_s"] > 0)
            gr.Markdown(f"Pre-rendered on all {len(rows)} VISO clips with aircraft or ships. The model never saw VISO "
                        f"in training. It warned in time on **{n_ok} of {len(rows)}**; the misses are listed too, "
                        "with the reason.")
            pick = gr.Dropdown(labels, value=labels[0] if labels else None, label="Clip")
            with gr.Row():
                vid = gr.Video(label="Tracked video", autoplay=True, loop=True)
                with gr.Column():
                    info = gr.Markdown()
                    tab = gr.Dataframe(headers=["track", "first alert frame", "time (s)", "P(entry)",
                                                "time to entry (s)"], label="First alert per track")
            pick.change(show_example, pick, [vid, info, tab])
            demo.load(show_example, pick, [vid, info, tab]) if labels else None
        with gr.Tab("Try your own clip"):
            gr.Markdown(f"Upload a short top-down video. The first {MAX_FRAMES} frames run on this free 2-core "
                        f"CPU, downscaled to a longest side of {MAX_SIDE} px. That takes about a minute. "
                        "Place the zone with the sliders.")
            with gr.Row():
                with gr.Column():
                    up = gr.Video(label="Clip", sources=["upload"])
                    zx = gr.Slider(0, 100, 50, step=1, label="Zone centre x (% of width)")
                    zy = gr.Slider(0, 100, 50, step=1, label="Zone centre y (% of height)")
                    zr = gr.Slider(10, 400, 60, step=5, label="Zone radius (px)")
                    go = gr.Button("Run TALOS", variant="primary")
                    prev = gr.Image(label="Zone preview (first frame)", interactive=False)
                with gr.Column():
                    out = gr.Video(label="Result")
                    note = gr.Markdown()
                    alerts = gr.Dataframe(headers=["track", "frame", "time (s)", "P(entry)", "time to entry (s)"],
                                          label="Alerts")
            for c in (up, zx, zy, zr):
                c.change(preview, [up, zx, zy, zr], prev)
            go.click(run_clip, [up, zx, zy, zr], [out, note, alerts], concurrency_limit=1)
        with gr.Tab("Results"):
            gr.Markdown(results_md())
        with gr.Tab("About"):
            gr.Markdown(ABOUT)
    return demo


demo = build()

if __name__ == "__main__":
    demo.queue(max_size=8).launch(theme=gr.themes.Base(primary_hue="red", neutral_hue="slate"))
