"""Smoke-test an assembled Space (scripts/build_space.sh output) the way the free CPU tier runs it.

    cd <built space dir> && python <repo>/scripts/test_space.py --clip-frames <frames dir>

Checks: every example loads, the Results tab renders the measured tables, the live path
runs on a real clip within the frame cap and returns a playable H.264 video, and the
server answers HTTP 200. Prints the live run's timing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, ".")
import app  # noqa: E402  (the Space's app.py in the current directory)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-frames", type=Path, required=True, help="folder of frames to turn into an upload")
    ap.add_argument("--zone", default="50,50,60", help="x%%,y%%,radius px")
    a = ap.parse_args()

    rows = app.example_rows()
    assert rows, "no examples in this build"
    for r in rows:
        vid, md, table = app.show_example(app.example_label(r))
        assert Path(vid).stat().st_size > 0 and "Result:" in md, r["seq"]
    print(f"examples ok: {len(rows)}")

    md = app.results_md()
    for must in ("TALOS-Track", "BoT-SORT", "AUROC", "speed-up"):
        assert must in md, must
    print("results tab ok")

    tmp = Path(tempfile.mkdtemp())
    up = tmp / "upload.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-i", str(a.clip_frames / "%06d.jpg"),
                    "-frames:v", "60", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(up)], check=True)
    zx, zy, zr = (float(v) for v in a.zone.split(","))
    t = time.time()
    out, note, alerts = app.run_clip(str(up), zx, zy, zr, progress=lambda *x, **k: None)
    dt = time.time() - t
    codec = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                            "-of", "csv=p=0", out], capture_output=True, text=True).stdout.strip()
    assert codec == "h264", codec
    assert dt < 120, f"live run took {dt:.0f} s on {app.THREADS} threads; the page promises about a minute"
    print(f"live run ok in {dt:.1f} s: {note}")

    th = threading.Thread(target=lambda: app.demo.launch(server_port=7861, prevent_thread_lock=True), daemon=True)
    th.start()
    for _ in range(60):
        try:
            code = urllib.request.urlopen("http://127.0.0.1:7861/", timeout=5).status
            break
        except Exception:
            time.sleep(1)
    else:
        raise SystemExit("server never answered")
    assert code == 200, code
    print("server ok: HTTP 200")
    print(json.dumps({"live_seconds": round(dt, 1), "note": note}))
    app.demo.close()


if __name__ == "__main__":
    main()
