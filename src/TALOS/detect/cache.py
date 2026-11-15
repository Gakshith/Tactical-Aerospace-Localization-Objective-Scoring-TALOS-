"""Detection cache: run a detector once, then evaluate every tracker on the
exact same boxes.

One ``.npy`` file per sequence holds an (N, 6) float32 array with rows
``[frame, x1, y1, x2, y2, conf]``. Frames are 1-based, to match MOT ground
truth. Layout::

    <root>/<detector_name>/<BENCH>/<seq>.npy
    <root>/<detector_name>/meta.json    (weights, imgsz, conf threshold, timing...)

``oracle`` is the detector name reserved for ground-truth boxes (conf 1),
which isolates association quality from detection quality.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.TALOS.data.mot import read_mot

COLS = ("frame", "x1", "y1", "x2", "y2", "conf")


def seq_path(root: Path, detector: str, bench: str, seq: str) -> Path:
    return Path(root) / detector / bench / f"{seq}.npy"


def save(root: Path, detector: str, bench: str, seq: str, dets: np.ndarray) -> Path:
    dets = np.asarray(dets, dtype=np.float32).reshape(-1, 6)
    if len(dets) and (dets[:, 0].min() < 1):
        raise ValueError("frames must be 1-based")
    p = seq_path(root, detector, bench, seq)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, dets[np.argsort(dets[:, 0], kind="stable")])
    return p


def load(root: Path, detector: str, bench: str, seq: str) -> np.ndarray:
    return np.load(seq_path(root, detector, bench, seq))


def by_frame(dets: np.ndarray, n_frames: int) -> list[np.ndarray]:
    """Split an (N, 6) cache into a list indexed 0..n_frames-1 of (k, 5)
    arrays ``[x1, y1, x2, y2, conf]``; frames with no boxes get (0, 5)."""
    out = [np.zeros((0, 5), dtype=np.float32) for _ in range(n_frames)]
    if len(dets) == 0:
        return out
    frames = dets[:, 0].astype(int)
    order = np.argsort(frames, kind="stable")
    frames, dets = frames[order], dets[order]
    starts = np.searchsorted(frames, np.arange(1, n_frames + 2))
    for f in range(n_frames):
        out[f] = dets[starts[f]:starts[f + 1], 1:6]
    return out


def write_meta(root: Path, detector: str, meta: dict) -> None:
    p = Path(root) / detector / "meta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2) + "\n")


def degrade(dets: np.ndarray, frame_wh: tuple[int, int], miss: float = 0.0,
            burst: int = 1, jitter: float = 0.0, fp_per_frame: float = 0.0,
            n_frames: int | None = None, seed: int = 0) -> np.ndarray:
    """Make ground-truth "detections" behave like a real detector, under control.

    miss          probability that an object is missed, applied in bursts of
                  ``burst`` consecutive frames per object (occlusion-like gaps)
    jitter        std of Gaussian noise on box centre and size, in px
    fp_per_frame  mean number of false-positive boxes per frame (Poisson),
                  sized like the true boxes and placed uniformly
    Confidences: true boxes ~ U(0.3, 1.0), false positives ~ U(0.1, 0.6), so
    low-score association (ByteTrack-style) has something to work with.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(dets, dtype=np.float64).copy()
    if len(d) == 0:
        return d.astype(np.float32)
    # object identity is lost in the cache, so bursts are drawn per box-row
    # neighbourhood: a miss at (frame f, row i) also removes the next burst-1
    # frames of the box nearest to it
    keep = np.ones(len(d), bool)
    if miss > 0:
        starts = rng.random(len(d)) < miss / max(burst, 1)
        centres = np.column_stack([(d[:, 1] + d[:, 3]) / 2, (d[:, 2] + d[:, 4]) / 2])
        frame = d[:, 0].astype(int)
        for i in np.flatnonzero(starts):
            for k in range(burst):
                rows = np.flatnonzero(frame == frame[i] + k)
                if len(rows):
                    j = rows[np.argmin(np.linalg.norm(centres[rows] - centres[i], axis=1))]
                    keep[j] = False
    d = d[keep]
    if jitter > 0 and len(d):
        w, h = d[:, 3] - d[:, 1], d[:, 4] - d[:, 2]
        cx = (d[:, 1] + d[:, 3]) / 2 + rng.normal(0, jitter, len(d))
        cy = (d[:, 2] + d[:, 4]) / 2 + rng.normal(0, jitter, len(d))
        w = np.maximum(2, w + rng.normal(0, jitter, len(d)))
        h = np.maximum(2, h + rng.normal(0, jitter, len(d)))
        d[:, 1:5] = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    d[:, 5] = rng.uniform(0.3, 1.0, len(d))
    if fp_per_frame > 0:
        W, H = frame_wh
        n_frames = n_frames or int(dets[:, 0].max())
        sizes = np.column_stack([dets[:, 3] - dets[:, 1], dets[:, 4] - dets[:, 2]])
        fps_rows = []
        for f in range(1, n_frames + 1):
            for _ in range(rng.poisson(fp_per_frame)):
                bw, bh = sizes[rng.integers(len(sizes))]
                x, y = rng.uniform(0, W - bw), rng.uniform(0, H - bh)
                fps_rows.append([f, x, y, x + bw, y + bh, rng.uniform(0.1, 0.6)])
        if fps_rows:
            d = np.vstack([d, fps_rows])
    return d[np.argsort(d[:, 0], kind="stable")].astype(np.float32)


def oracle_from_gt(gt_txt: Path) -> np.ndarray:
    """Ground-truth boxes as detections with conf 1."""
    gt = read_mot(gt_txt)
    if len(gt) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    x, y, w, h = gt[:, 2], gt[:, 3], gt[:, 4], gt[:, 5]
    return np.column_stack([gt[:, 0], x, y, x + w, y + h, np.ones(len(gt))]).astype(np.float32)
