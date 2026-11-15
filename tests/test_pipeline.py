"""End-to-end pipeline on a synthetic video with a fake detector (no GPU, no weights)."""
import json

import cv2
import numpy as np

from src.TALOS.pipeline.run import Pipeline, run_video
from src.TALOS.threat.model import Zone


class FakeDetector:
    """Reports two objects: a plane flying at the zone and a parked ship."""
    def __init__(self):
        self.f = 0

    def predict(self, frames):
        out = []
        for _ in frames:
            self.f += 1
            x = 40 + 6 * self.f                       # 6 px/frame = 60 px/s at 10 fps
            out.append(np.array([[x, 200, x + 24, 224, 0.9, 0],
                                 [500, 400, 516, 412, 0.8, 1]], float))
        return out


def _video(path, n=60, w=640, h=480):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for _ in range(n):
        vw.write(np.full((h, w, 3), 60, np.uint8))
    vw.release()


def test_pipeline_end_to_end(tmp_path):
    _video(tmp_path / "in.mp4")
    pipe = Pipeline(FakeDetector(), fps=10, zone=Zone.circle(560, 212, 40), threat_every=1)
    timing = run_video(tmp_path / "in.mp4", tmp_path / "out", pipe, batch=4)
    assert timing["frames"] == 60
    assert set(timing["ms_per_frame"]) >= {"decode", "detect", "track", "threat", "render", "total"}
    assert (tmp_path / "out" / "annotated.mp4").stat().st_size > 0

    rows = (tmp_path / "out" / "tracks.csv").read_text().splitlines()
    ids = {(r.split(",")[1], r.split(",")[2]) for r in rows[1:]}
    assert ("plane", "1") in ids and ("ship", "1") in ids   # one stable track per object

    alerts = [json.loads(x) for x in (tmp_path / "out" / "alerts.jsonl").read_text().splitlines()]
    assert alerts, "the plane heading straight at the zone must raise an alert"
    assert {a["class"] for a in alerts} == {"plane"}        # the parked ship never alerts
    first = min(a["frame"] for a in alerts)
    # the plane reaches the zone edge (x = 520) near frame 80, so the first alert
    # must come well before the end of the clip
    assert first < 50


def test_no_zone_means_no_alerts(tmp_path):
    _video(tmp_path / "in.mp4", n=20)
    pipe = Pipeline(FakeDetector(), fps=10, zone=None)
    run_video(tmp_path / "in.mp4", tmp_path / "out", pipe, write_video=False)
    assert (tmp_path / "out" / "alerts.jsonl").read_text() == ""


def test_frame_folder_input(tmp_path):
    d = tmp_path / "img1"
    d.mkdir()
    for i in range(1, 13):
        cv2.imwrite(str(d / f"{i:06d}.jpg"), np.full((240, 320, 3), 50, np.uint8))
    pipe = Pipeline(FakeDetector(), fps=10, zone=None)
    timing = run_video(d, tmp_path / "out", pipe, write_video=False)
    assert timing["frames"] == 12
