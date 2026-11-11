"""Draw MOT-format boxes and IDs onto a sequence's frames and write an mp4.

Used to eyeball converted ground truth (does every box sit on an object, do
IDs stay put?) and later to render tracker output the same way.

    python -m src.TALOS.data.overlay <seq_dir> <mot_txt> <out.mp4> [--crop x0 y0 x1 y1] [--max-frames N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.TALOS.data.mot import read_mot, read_seqinfo


def id_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(track_id) * 7919)
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


def render(seq_dir: Path, mot_txt: Path, out_path: Path,
           crop: tuple[int, int, int, int] | None = None,
           max_frames: int | None = None, max_width: int = 1600) -> int:
    info = read_seqinfo(seq_dir)
    rows = read_mot(mot_txt)
    by_frame: dict[int, np.ndarray] = {}
    for f in np.unique(rows[:, 0]).astype(int):
        by_frame[f] = rows[rows[:, 0] == f]

    img_dir = Path(seq_dir) / info.img_dir
    frames = sorted(img_dir.glob(f"*{info.img_ext}"))
    if max_frames:
        frames = frames[:max_frames]
    writer = None
    for i, fp in enumerate(frames, start=1):
        img = cv2.imread(str(fp))
        if img is None:
            raise FileNotFoundError(fp)
        for r in by_frame.get(i, ()):
            x, y, w, h = r[2:6]
            c = id_color(int(r[1]))
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), c, 2)
            cv2.putText(img, str(int(r[1])), (int(x), int(y) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        if crop:
            x0, y0, x1, y1 = crop
            img = img[y0:y1, x0:x1]
        if img.shape[1] > max_width:
            s = max_width / img.shape[1]
            img = cv2.resize(img, (max_width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        if writer is None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     max(info.fps, 1.0), (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer is not None:
        writer.release()
    return len(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir", type=Path)
    ap.add_argument("mot_txt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--crop", type=int, nargs=4)
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args()
    n = render(a.seq_dir, a.mot_txt, a.out, tuple(a.crop) if a.crop else None, a.max_frames)
    print(f"wrote {n} frames to {a.out}")


if __name__ == "__main__":
    main()
