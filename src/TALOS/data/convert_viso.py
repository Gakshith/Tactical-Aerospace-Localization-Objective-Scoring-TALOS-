"""Convert VISO's MOT subset (planes and ships) into MOTChallenge benchmarks.

VISO is the cross-dataset check: nothing in TALOS is trained or tuned on it.
Source layout (measured, see docs/data/viso.md)::

    mot/<class>/<seq>/img/000001.jpg
    mot/<class>/<seq>/gt/gt.txt

Two label formats coexist, detected per file:
  A  comma-separated  frame,id,x_left,y_top,w,h,...
  B  space-separated  frame id x1 y1 x2 y2 ...   (CRLF; corners, inferred:
                      read as w/h, planes would be ~1,300 px wide)

Four (frame, id) pairs appear twice, one row far from the track (a
mislabelled id). For each, the row nearest the track's position in the
neighbouring frames is kept and the other dropped; every drop is reported.

    python -m src.TALOS.data.convert_viso <viso mot dir> <out dir>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.TALOS.data.mot import SeqInfo, write_mot, write_seqinfo

CLASSES = {"plane": "VISO_plane", "ship": "VISO_ship"}
FPS = 10.0  # stated in the VISO paper (arXiv 2111.12960, Sec. III-A1)


def load_gt(path: Path) -> tuple[np.ndarray, str]:
    text = path.read_text().replace("\r", "").strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if "," in lines[0]:
        rows = np.array([[float(v) for v in ln.split(",")[:6]] for ln in lines])
        return rows, "A"
    raw = np.array([[float(v) for v in ln.split()[:6]] for ln in lines])
    x1, y1, x2, y2 = raw[:, 2], raw[:, 3], raw[:, 4], raw[:, 5]
    if not ((x2 > x1) & (y2 > y1)).all():
        raise ValueError(f"{path}: format B rows are not x1<x2, y1<y2")
    return np.column_stack([raw[:, 0], raw[:, 1], x1, y1, x2 - x1, y2 - y1]), "B"


def repair_duplicates(rows: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Keep, for each duplicated (frame, id), the row closest to the track's
    centre in the nearest other frames."""
    dropped = []
    keep = np.ones(len(rows), bool)
    keys = rows[:, :2]
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    for f, tid in uniq[counts > 1]:
        idx = np.flatnonzero((rows[:, 0] == f) & (rows[:, 1] == tid))
        others = rows[(rows[:, 1] == tid) & (rows[:, 0] != f)]
        near = others[np.argsort(np.abs(others[:, 0] - f))[:4]]
        ref = (near[:, 2:4] + near[:, 4:6] / 2).mean(axis=0)
        centres = rows[idx, 2:4] + rows[idx, 4:6] / 2
        best = idx[np.argmin(np.linalg.norm(centres - ref, axis=1))]
        for i in idx:
            if i != best:
                keep[i] = False
                dropped.append({"frame": int(f), "id": int(tid),
                                "dropped_xywh": [float(v) for v in rows[i, 2:6]]})
    return rows[keep], dropped


def convert(src: Path, out: Path) -> dict:
    report = {"fps": FPS, "sequences": {}, "dropped_rows": {}}
    for cls, bench in CLASSES.items():
        report["sequences"][bench] = []
        for sdir in sorted((src / cls).iterdir()):
            if not (sdir / "gt" / "gt.txt").exists():
                continue
            rows, fmt = load_gt(sdir / "gt" / "gt.txt")
            rows, dropped = repair_duplicates(rows)
            frames = sorted((sdir / "img").glob("*.jpg"))
            if [int(f.stem) for f in frames] != list(range(1, len(frames) + 1)):
                raise ValueError(f"{sdir}: frames are not 1..N contiguous")
            if rows[:, 0].max() > len(frames):
                raise ValueError(f"{sdir}: GT frame beyond last image")
            h, w = cv2.imread(str(frames[0])).shape[:2]
            seq = f"viso-{sdir.name}"
            odir = out / f"{bench}-all" / seq
            mot = np.column_stack([rows, np.ones((len(rows), 3))])  # conf, class, vis = 1
            write_mot(odir / "gt" / "gt.txt", mot)
            write_seqinfo(odir, SeqInfo(seq, len(frames), w, h, FPS, "img1", ".jpg"))
            link = odir / "img1"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink((sdir / "img").resolve(), link)
            report["sequences"][bench].append({"seq": seq, "format": fmt, "frames": len(frames),
                                               "size": [w, h], "tracks": int(len(np.unique(rows[:, 1]))),
                                               "boxes": int(len(rows))})
            if dropped:
                report["dropped_rows"][seq] = dropped
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="VISO mot/ directory")
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    report = convert(a.src, a.out)
    (a.out / "viso_convert_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for bench, seqs in report["sequences"].items():
        print(f"{bench}: {len(seqs)} seqs, {sum(s['tracks'] for s in seqs)} tracks, "
              f"{sum(s['boxes'] for s in seqs)} boxes")
    print("dropped:", {k: len(v) for k, v in report["dropped_rows"].items()})


if __name__ == "__main__":
    main()
