"""Frame-level detection quality from a detection cache against MOT ground truth.

Matching is COCO/VOC style, per frame: detections in descending confidence
each take the unmatched GT box with the highest IoU, if that IoU >= 0.5.
Because matching is greedy in confidence order, the boxes kept at a higher
threshold match exactly as they would alone, so one matching over all cached
boxes gives precision/recall at every threshold.

    python -m src.TALOS.eval.det_metrics --dets-root ~/talos/data/dets --detector v1_cgi \
        --bench AIRMOT_plane --bench-dir ~/talos/data/mot/AIRMOT_plane-all \
        --splits val test --conf 0.5 --out ~/talos/results/det_v1_cgi/metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.TALOS.data.mot import read_mot, read_seqinfo
from src.TALOS.detect import cache

SIZE_BINS = (("lt16", 0, 16), ("16to32", 16, 32), ("gt32", 32, np.inf))  # by max(w, h), px


def size_bin(wh_max: np.ndarray) -> np.ndarray:
    """0: < 16 px, 1: 16..32 px inclusive, 2: > 32 px."""
    return np.where(wh_max < 16, 0, np.where(wh_max <= 32, 1, 2))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (n,4) and (m,4) xyxy boxes."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_frame(dets: np.ndarray, gts: np.ndarray, iou_thr: float = 0.5):
    """dets (k,5) [x1,y1,x2,y2,conf], gts (m,4) xyxy.
    Returns (tp flag per det, conf of the det that matched each GT or -1)."""
    tp = np.zeros(len(dets), bool)
    gt_conf = np.full(len(gts), -1.0)
    if len(dets) == 0 or len(gts) == 0:
        return tp, gt_conf
    order = np.argsort(-dets[:, 4], kind="stable")
    ious = iou_matrix(dets[:, :4], gts)
    taken = np.zeros(len(gts), bool)
    for i in order:
        cand = np.where(taken, -1.0, ious[i])
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            taken[j] = True
            tp[i] = True
            gt_conf[j] = dets[i, 4]
    return tp, gt_conf


def ap_all_point(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    """Area under the monotone (interpolated) precision envelope, all points (VOC2010+/COCO-style)."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-conf, kind="stable")
    tpc = np.cumsum(tp[order])
    fpc = np.cumsum(~tp[order])
    rec = tpc / n_gt
    prec = tpc / (tpc + fpc)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def match_sequence(dets: np.ndarray, gt: np.ndarray, n_frames: int, min_conf: float):
    """dets: cache rows (N,6); gt: MOT rows (M,>=6). Returns arrays over all
    dets (conf, tp) and over all GT boxes (matched-det conf, size bin)."""
    dets = dets[dets[:, 5] >= min_conf] if len(dets) else dets
    per_f = cache.by_frame(dets, n_frames)
    gtf = gt[:, 0].astype(int) if len(gt) else np.zeros(0, int)
    gxyxy = np.column_stack([gt[:, 2], gt[:, 3], gt[:, 2] + gt[:, 4], gt[:, 3] + gt[:, 5]]) if len(gt) else np.zeros((0, 4))
    gsize = size_bin(np.maximum(gt[:, 4], gt[:, 5])) if len(gt) else np.zeros(0, int)
    confs, tps, gconf, gbins = [], [], [], []
    for f in range(1, n_frames + 1):
        d = per_f[f - 1]
        m = gtf == f
        tp, gc = match_frame(d, gxyxy[m])
        confs.append(d[:, 4]); tps.append(tp); gconf.append(gc); gbins.append(gsize[m])
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)  # noqa: E731
    return cat(confs, float), cat(tps, bool), cat(gconf, float), cat(gbins, int)


def summarize(conf, tp, gconf, gbins, at_conf: float, extra_confs=(0.1, 0.25, 0.5)) -> dict:
    n_gt = len(gconf)

    def pr(c):
        keep = conf >= c
        ntp, nd = int(tp[keep].sum()), int(keep.sum())
        p = ntp / nd if nd else float("nan")
        r = ntp / n_gt if n_gt else float("nan")
        f1 = 2 * p * r / (p + r) if nd and n_gt and (p + r) > 0 else 0.0
        return {"dets": nd, "tp": ntp, "fp": nd - ntp, "fn": n_gt - ntp,
                "precision": p, "recall": r, "f1": f1}

    out = {"gt_boxes": n_gt, "dets_all": int(len(conf)), f"at_conf_{at_conf:g}": pr(at_conf),
           "ap50": ap_all_point(tp, conf, n_gt),
           "pr_by_conf": {f"{c:g}": pr(c) for c in sorted(set(extra_confs) | {at_conf})}}
    by_size = {}
    for b, (name, lo, hi) in enumerate(SIZE_BINS):
        m = gbins == b
        n = int(m.sum())
        by_size[name] = {"gt_boxes": n,
                         f"recall_at_{at_conf:g}": float((gconf[m] >= at_conf).sum() / n) if n else float("nan"),
                         "recall_at_min_conf": float((gconf[m] >= 0).sum() / n) if n else float("nan")}
    out["recall_by_size"] = by_size
    return out


def evaluate_split(dets_root, detector, bench, bench_dir, seqs, at_conf, min_conf) -> dict:
    agg = [[], [], [], []]
    per_seq = {}
    frames = 0
    for s in seqs:
        info = read_seqinfo(bench_dir / s)
        gt = read_mot(bench_dir / s / "gt" / "gt.txt")
        dets = cache.load(dets_root, detector, bench, s)
        r = match_sequence(dets, gt, info.length, min_conf)
        for a, x in zip(agg, r):
            a.append(x)
        frames += info.length
        sm = summarize(*r, at_conf=at_conf)
        per_seq[s] = {"frames": info.length, "gt_boxes": sm["gt_boxes"], "ap50": sm["ap50"],
                      **{k: sm[f"at_conf_{at_conf:g}"][k] for k in ("dets", "precision", "recall")}}
    tot = summarize(*(np.concatenate(a) for a in agg), at_conf=at_conf)
    return {"seqs": list(seqs), "frames": frames, **tot, "per_seq": per_seq}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return None if np.isnan(o) else round(o, 5)
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets-root", type=Path, required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--split-file", default="configs/splits/airmot.json")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--conf", type=float, default=None, help="operating threshold; default meta v1_conf")
    ap.add_argument("--min-conf", type=float, default=0.1, help="boxes used for AP")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    meta_p = a.dets_root / a.detector / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    conf = a.conf if a.conf is not None else meta.get("v1_conf", 0.5)
    split = json.loads(Path(a.split_file).read_text())["split"]
    present = {p.name for p in a.bench_dir.iterdir() if (p / "seqinfo.ini").exists()}

    res = {"detector": a.detector, "bench": a.bench, "iou_thr": 0.5, "conf": conf, "min_conf_for_ap": a.min_conf,
           "size_bins_px_maxwh": {n: [lo, None if hi == np.inf else hi] for n, lo, hi in SIZE_BINS},
           "matching": "per frame, greedy by conf, IoU>=0.5; AP all-point interpolated",
           "detector_meta": {k: meta.get(k) for k in ("weights", "weights_sha256", "imgsz_used", "save_conf",
                                                       "v1_conf", "ultralytics", "torch", "gpu")},
           "splits": {}}
    for sp in a.splits:
        seqs = [s for s in sorted(split[sp]) if s in present]
        res["splits"][sp] = evaluate_split(a.dets_root, a.detector, a.bench, a.bench_dir, seqs, conf, a.min_conf)
        r = res["splits"][sp]
        at = r[f"at_conf_{conf:g}"]
        print(f"{a.bench} {sp}: {len(seqs)} seqs {r['frames']} frames {r['gt_boxes']} GT | conf>={conf:g}: "
              f"{at['dets']} dets P={at['precision']:.3f} R={at['recall']:.3f} | AP50={r['ap50']:.3f} | "
              + " ".join(f"{k}:{v[f'recall_at_{conf:g}']}" for k, v in r["recall_by_size"].items()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(a.out.read_text()) if a.out.exists() else {}
    existing[a.bench] = _clean(res)
    a.out.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
