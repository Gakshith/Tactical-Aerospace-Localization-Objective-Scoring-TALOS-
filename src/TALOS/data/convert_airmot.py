"""Convert AIR-MOT-100 into per-class MOTChallenge benchmarks.

Source layout (measured, see docs/data/airmot.md)::

    AIR-MOT-100/<n>/img/000001.jpg      (videos 95-100: 000001_<k>.jpg)
    AIR-MOT-100/<n>/gt/gt.txt           frame,id,x,y,w,h,1,class,1   class 1=aircraft 2=ship

Output, one benchmark per class (TrackEval evaluates one class at a time)::

    <out>/AIRMOT_plane-all/<seq>/{gt/gt.txt, seqinfo.ini, img1 -> source img dir}
    <out>/AIRMOT_ship-all/<seq>/...

Only videos that contain the class get a sequence. Unannotated videos (empty
gt.txt) are skipped and listed. Exact duplicate rows are dropped. Track IDs
in the source are unique only within a class, which per-class output keeps.

    python -m src.TALOS.data.convert_airmot <AIR-MOT-100 dir> <out dir> [--fps 10]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.TALOS.data.mot import SeqInfo, write_mot, write_seqinfo

CLASSES = {1: "plane", 2: "ship"}


def load_gt(path: Path) -> np.ndarray:
    text = path.read_text().replace("\r", "").strip()
    if not text:
        return np.zeros((0, 9))
    rows = np.array([[float(v) for v in line.split(",")[:9]] for line in text.splitlines()])
    return np.unique(rows, axis=0)  # drops the exact duplicate rows found in v12


def convert(src: Path, out: Path, fps: float) -> dict:
    report = {"fps_assumed": fps, "skipped_unannotated": [], "sequences": {c: [] for c in CLASSES.values()}}
    videos = sorted((p for p in src.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    for vdir in videos:
        gt = load_gt(vdir / "gt" / "gt.txt")
        if len(gt) == 0:
            report["skipped_unannotated"].append(vdir.name)
            continue
        frames = sorted((vdir / "img").glob("*.jpg"))
        first = cv2.imread(str(frames[0]))
        h, w = first.shape[:2]
        # frame number = leading digits, before any "_k" suffix
        numbers = [int(f.stem.split("_")[0]) for f in frames]
        if numbers != list(range(1, len(frames) + 1)):
            raise ValueError(f"video {vdir.name}: frames are not 1..N contiguous")
        ext = "".join(frames[0].suffixes)
        suffix = frames[0].stem[len(frames[0].stem.split("_")[0]):]  # "" or "_k"

        for cls_id, cls_name in CLASSES.items():
            rows = gt[gt[:, 7] == cls_id].copy()
            if len(rows) == 0:
                continue
            if rows[:, 0].max() > len(frames):
                raise ValueError(f"video {vdir.name}: GT frame beyond last image")
            rows[:, 6] = 1  # evaluate every box
            rows[:, 7] = 1  # TrackEval's evaluated class id
            rows[:, 8] = 1
            seq = f"airmot-{int(vdir.name):03d}"
            sdir = out / f"AIRMOT_{cls_name}-all" / seq
            write_mot(sdir / "gt" / "gt.txt", rows)
            write_seqinfo(sdir, SeqInfo(seq, len(frames), w, h, fps, "img1", suffix + ext))
            link = sdir / "img1"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink((vdir / "img").resolve(), link)
            report["sequences"][cls_name].append({
                "seq": seq, "frames": len(frames), "tracks": int(len(np.unique(rows[:, 1]))),
                "boxes": int(len(rows)),
            })
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fps", type=float, default=10.0,
                    help="not stated by the dataset; Jilin-1 video is commonly cited at 10 fps")
    a = ap.parse_args()
    report = convert(a.src, a.out, a.fps)
    (a.out / "airmot_convert_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for cls, seqs in report["sequences"].items():
        print(f"{cls}: {len(seqs)} sequences, {sum(s['tracks'] for s in seqs)} tracks, "
              f"{sum(s['boxes'] for s in seqs)} boxes")
    print(f"skipped unannotated: {len(report['skipped_unannotated'])}")


if __name__ == "__main__":
    main()
