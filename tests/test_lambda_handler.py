"""The Lambda handler (deploy/lambda/handler.py) with a fake detector: routing, limits, zones."""
import base64
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest



class FakeDetector:
    """A plane flying right at 6 px/frame and a parked ship (same as tests/test_pipeline.py)."""
    def __init__(self):
        self.f = 0

    def predict(self, frames):
        out = []
        for _ in frames:
            self.f += 1
            x = 40 + 6 * self.f
            out.append(np.array([[x, 200, x + 24, 224, 0.9, 0], [500, 400, 516, 412, 0.8, 1]], float))
        return out


spec = importlib.util.spec_from_file_location(
    "talos_lambda", Path(__file__).resolve().parents[1] / "deploy" / "lambda" / "handler.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def _clip(tmp_path, n=50, w=640, h=480) -> str:
    p = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for _ in range(n):
        vw.write(np.full((h, w, 3), 60, np.uint8))
    vw.release()
    return base64.b64encode(p.read_bytes()).decode()


def _url_event(method, body=None):
    return {"requestContext": {"http": {"method": method}}, "body": json.dumps(body) if body else "",
            "isBase64Encoded": False}


@pytest.fixture(autouse=True)
def fake_detector(monkeypatch):
    monkeypatch.setattr(H, "_detector", FakeDetector())


def test_get_returns_usage():
    r = H.handler(_url_event("GET"))
    assert r["statusCode"] == 200 and "limits" in json.loads(r["body"])


def test_missing_input_is_400():
    r = H.handler(_url_event("POST", {"fps": 10}))
    assert r["statusCode"] == 400


def test_bad_json_is_400():
    ev = _url_event("POST")
    ev["body"] = "{not json"
    assert H.handler(ev)["statusCode"] == 400


def test_video_tracks_and_alerts(tmp_path):
    body = {"video_b64": _clip(tmp_path), "zone_circle": [300, 212, 40], "fps": 10, "return_frame": True}
    r = H.handler(_url_event("POST", body))
    out = json.loads(r["body"])
    assert r["statusCode"] == 200, out
    assert out["frames"] == H.MAX_FRAMES                     # 50-frame clip capped at 40
    classes = {t["class"] for t in out["tracks"]}
    assert classes == {"plane", "ship"}
    assert any(a["class"] == "plane" for a in out["alerts"])  # the plane flies into the zone
    assert not any(a["class"] == "ship" for a in out["alerts"])
    assert base64.b64decode(out["last_frame_jpeg_b64"])[:2] == b"\xff\xd8"


def test_direct_invoke_single_image():
    ok, jpg = cv2.imencode(".jpg", np.full((480, 640, 3), 60, np.uint8))
    r = H.handler({"image_b64": base64.b64encode(jpg.tobytes()).decode()})
    out = json.loads(r["body"])
    assert r["statusCode"] == 200 and out["frames"] == 1 and out["alerts"] == []


def test_large_frames_are_downscaled_and_zone_follows(tmp_path):
    ok, jpg = cv2.imencode(".jpg", np.full((1440, 2560, 3), 60, np.uint8))
    out = H.process({"image_b64": base64.b64encode(jpg.tobytes()).decode(), "zone_circle": [1280, 720, 100]},
                    det=FakeDetector())
    assert out["processed_scale"] == 0.5 and out["input_size"] == [2560, 1440]
