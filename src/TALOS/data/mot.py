"""MOTChallenge text format: the one interchange format between converters,
trackers and TrackEval.

Each row is ``frame, id, x, y, w, h, conf, class, visibility`` with 1-based
frames and top-left ``(x, y)`` box corners in pixels. Ground truth uses
``conf = 1`` for "evaluate this box"; tracker output uses the detection score.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLUMNS = ("frame", "id", "x", "y", "w", "h", "conf", "cls", "vis")


@dataclass(frozen=True)
class SeqInfo:
    name: str
    length: int  # number of frames
    width: int
    height: int
    fps: float
    img_dir: str = "img1"
    img_ext: str = ".jpg"


def read_mot(path: str | Path) -> np.ndarray:
    """Read a MOT txt file into an (N, 9) float array. Missing trailing
    columns are filled with 1 (conf, class, visibility)."""
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    if rows.size == 0:
        return np.zeros((0, 9))
    if rows.shape[1] < 9:
        pad = np.ones((rows.shape[0], 9 - rows.shape[1]))
        rows = np.hstack([rows, pad])
    return rows[:, :9]


def write_mot(path: str | Path, rows: np.ndarray) -> None:
    """Write an (N, >=6) array as MOT txt, sorted by frame then id."""
    rows = np.asarray(rows, dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows.size == 0:
        path.write_text("")
        return
    if rows.shape[1] < 9:
        rows = np.hstack([rows, np.ones((rows.shape[0], 9 - rows.shape[1]))])
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    with path.open("w") as f:
        for r in rows:
            f.write(
                f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},"
                f"{r[5]:.2f},{r[6]:.4g},{int(r[7])},{r[8]:.4g}\n"
            )


def write_seqinfo(seq_dir: str | Path, info: SeqInfo) -> None:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # TrackEval reads these keys case-sensitively
    cfg["Sequence"] = {
        "name": info.name,
        "imDir": info.img_dir,
        "frameRate": f"{info.fps:g}",
        "seqLength": str(info.length),
        "imWidth": str(info.width),
        "imHeight": str(info.height),
        "imExt": info.img_ext,
    }
    seq_dir = Path(seq_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)
    with (seq_dir / "seqinfo.ini").open("w") as f:
        cfg.write(f, space_around_delimiters=False)


def read_seqinfo(seq_dir: str | Path) -> SeqInfo:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(Path(seq_dir) / "seqinfo.ini")
    s = cfg["Sequence"]
    return SeqInfo(
        name=s["name"],
        length=int(s["seqLength"]),
        width=int(s["imWidth"]),
        height=int(s["imHeight"]),
        fps=float(s["frameRate"]),
        img_dir=s.get("imDir", "img1"),
        img_ext=s.get("imExt", ".jpg"),
    )


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.column_stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]])


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])
