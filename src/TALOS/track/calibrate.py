"""Estimate the Kalman measurement noise R from a detector's own errors.

A tracker's gate is only as good as its noise model: assume 1 px of noise
for a detector that jitters 2 px and the Mahalanobis gate rejects the
object's own detections, so tracks fragment. Instead of hand-tuning, match
the detector's boxes to ground truth (IoU >= 0.5) on a split, and fit the
residual spread as a function of object size:

    sigma(size) = max(sigma_min, frac * size)

The output plugs straight into TALOS-Track's ``meas_pos_min`` and
``meas_pos_frac``.

    python -m src.TALOS.track.calibrate --bench AIRMOT_plane --split val --detector <name>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.TALOS.data.mot import read_mot, read_seqinfo


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def residuals(dets: np.ndarray, gt: np.ndarray, min_conf: float = 0.0,
              iou_min: float = 0.5) -> np.ndarray:
    """Rows of [d_cx, d_cy, d_w, d_h, gt_size] for matched detection/GT pairs.
    ``dets`` is a cache array (frame, x1, y1, x2, y2, conf); ``gt`` MOT rows."""
    out = []
    dets = dets[dets[:, 5] >= min_conf]
    gxy = np.column_stack([gt[:, 0], gt[:, 2], gt[:, 3], gt[:, 2] + gt[:, 4], gt[:, 3] + gt[:, 5]])
    for f in np.unique(gxy[:, 0]):
        g = gxy[gxy[:, 0] == f, 1:]
        d = dets[dets[:, 0] == f, 1:5]
        if not len(d):
            continue
        iou = iou_matrix(d, g)
        r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c):
            if iou[i, j] < iou_min:
                continue
            dc = (d[i, :2] + d[i, 2:]) / 2 - (g[j, :2] + g[j, 2:]) / 2
            dwh = (d[i, 2:] - d[i, :2]) - (g[j, 2:] - g[j, :2])
            out.append([*dc, *dwh, max(g[j, 2] - g[j, 0], g[j, 3] - g[j, 1])])
    return np.array(out).reshape(-1, 5)


def fit(res: np.ndarray, n_bins: int = 6) -> dict:
    """Robust per-size-bin std (1.4826 x MAD, pooled over the 4 components),
    then sigma(size) = max(sigma_min, frac * size) by least squares on bins."""
    if len(res) < 50:
        raise ValueError(f"only {len(res)} matched boxes; not enough to calibrate")
    size = res[:, 4]
    edges = np.unique(np.quantile(size, np.linspace(0, 1, n_bins + 1)))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (size >= lo) & (size <= hi)
        if m.sum() < 20:
            continue
        r = res[m, :4]
        mad = np.median(np.abs(r - np.median(r, axis=0)), axis=0) * 1.4826
        bins.append((float(np.median(size[m])), float(np.sqrt(np.mean(mad ** 2))), int(m.sum())))
    s = np.array([b[0] for b in bins])
    sig = np.array([b[1] for b in bins])
    frac = float(np.sum(s * sig) / np.sum(s * s))       # sigma ~ frac * size through the origin
    sigma_min = float(np.min(sig))
    bias = np.median(res[:, :4], axis=0)
    return {"meas_pos_min": round(sigma_min, 3), "meas_pos_frac": round(frac, 4),
            "bins": [{"size_px": b[0], "sigma_px": round(b[1], 3), "n": b[2]} for b in bins],
            "median_bias_cx_cy_w_h": [round(float(v), 3) for v in bias], "n_matched": int(len(res))}


def main() -> None:
    import os
    from src.TALOS.data.split import load_split
    from src.TALOS.detect import cache

    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])  # never test
    ap.add_argument("--detector", required=True)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    data = Path(os.environ.get("DATA", Path.home() / "talos" / "data"))
    det_root = data / "dets"
    res = []
    for s in load_split("airmot")[a.split]:
        sdir = data / "mot" / f"{a.bench}-all" / s
        if not sdir.is_dir() or not cache.seq_path(det_root, a.detector, a.bench, s).exists():
            continue
        read_seqinfo(sdir)
        res.append(residuals(cache.load(det_root, a.detector, a.bench, s),
                             read_mot(sdir / "gt" / "gt.txt"), a.min_conf))
    result = {"bench": a.bench, "split": a.split, "detector": a.detector, **fit(np.vstack(res))}
    text = json.dumps(result, indent=2)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
