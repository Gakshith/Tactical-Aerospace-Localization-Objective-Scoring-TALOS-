"""Benchmark trackers on cached detections and score them with TrackEval.

    python -m src.TALOS.eval.bench --bench AIRMOT_plane --split val \
        --detector oracle --trackers v1 sort bytetrack ocsort botsort \
        --run-id d2-oracle-plane-val

Writes, under ``results/<run_id>/``:
    metrics.json            per-tracker HOTA/IDF1/MOTA/... + timing + provenance
    trk/<BENCH>-<split>/<tracker>/data/<seq>.txt   MOT output (for TrackEval and overlays)

The detector ``oracle`` is built on the fly from ground truth: it scores
association alone, with detection made perfect.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
from pathlib import Path

import numpy as np

from src.TALOS.data.mot import read_seqinfo, write_mot
from src.TALOS.data.split import load_split
from src.TALOS.detect import cache as det_cache
from src.TALOS.eval.trackeval_runner import run_metadata, run_trackeval
from src.TALOS.track.base import frames_from_cache, get, run_sequence

DATA = Path(os.environ.get("DATA", Path.home() / "talos" / "data"))


def bench_seqs(bench: str, split: str, dataset: str = "airmot") -> list[str]:
    """Sequences of this split that exist for this benchmark's class."""
    names = load_split(dataset)[split]
    root = DATA / "mot" / f"{bench}-all"
    return [s for s in names if (root / s).is_dir()]


NOISY_ORACLE = {"miss": 0.2, "burst": 5, "jitter": 2.0, "fp_per_frame": 0.5, "seed": 0}


def ensure_oracle(bench: str, seqs: list[str], det_root: Path, name: str) -> None:
    """``oracle``: GT boxes. ``oracle-noisy``: GT degraded with NOISY_ORACLE
    (20% misses in 5-frame bursts, 2 px jitter, 0.5 false positives/frame)."""
    for s in seqs:
        if det_cache.seq_path(det_root, name, bench, s).exists():
            continue
        sdir = DATA / "mot" / f"{bench}-all" / s
        dets = det_cache.oracle_from_gt(sdir / "gt" / "gt.txt")
        if name == "oracle-noisy":
            info = read_seqinfo(sdir)
            dets = det_cache.degrade(dets, (info.width, info.height), n_frames=info.length,
                                     **NOISY_ORACLE)
        det_cache.save(det_root, name, bench, s, dets)
    if name == "oracle-noisy":
        det_cache.write_meta(det_root, name, {"from": "ground truth", **NOISY_ORACLE})


def stage_gt(bench: str, split: str, seqs: list[str], gt_root: Path) -> None:
    """Expose <bench>-all sequences under TrackEval's <bench>-<split> folder name."""
    dst = gt_root / f"{bench}-{split}"
    dst.mkdir(parents=True, exist_ok=True)
    for s in seqs:
        link = dst / s
        if not link.exists():
            os.symlink(DATA / "mot" / f"{bench}-all" / s, link)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--detector", required=True)
    ap.add_argument("--trackers", nargs="+", required=True)
    ap.add_argument("--params", default="{}", help='JSON {tracker: {param: value}}')
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--det-root", type=Path, default=DATA / "dets")
    ap.add_argument("--out-root", type=Path, default=Path("results"))
    a = ap.parse_args()

    if a.split == "test":
        log = Path("results/TEST_LOG.md")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write(f"- {a.run_id}: {a.bench} test, detector={a.detector}, trackers={a.trackers}\n")

    params = json.loads(a.params)
    seqs = bench_seqs(a.bench, a.split)
    if not seqs:
        raise SystemExit(f"no {a.bench} sequences in split {a.split}")
    if a.detector in ("oracle", "oracle-noisy"):
        ensure_oracle(a.bench, seqs, a.det_root, a.detector)

    out = a.out_root / a.run_id
    if out.exists():
        raise SystemExit(f"{out} exists; run ids are never reused")
    gt_root = out / "gt"
    stage_gt(a.bench, a.split, seqs, gt_root)
    trk_root = out / "trk"

    timing = {}
    for name in a.trackers:
        factory = get(name)
        total_s, total_f = 0.0, 0
        for s in seqs:
            info = read_seqinfo(DATA / "mot" / f"{a.bench}-all" / s)
            frames = frames_from_cache(a.det_root, a.detector, a.bench, s, info.length, a.min_conf)
            res = run_sequence(factory(fps=info.fps, **params.get(name, {})), frames)
            write_mot(trk_root / f"{a.bench}-{a.split}" / name / "data" / f"{s}.txt", res.rows)
            total_s += res.seconds
            total_f += res.n_frames
        timing[name] = {"frames": total_f, "track_seconds": round(total_s, 4),
                        "track_ms_per_frame": round(1000 * total_s / max(total_f, 1), 4)}

    scores = run_trackeval(gt_root, trk_root, a.bench, a.split, a.trackers, seqs, num_cores=8)
    config = {"bench": a.bench, "split": a.split, "detector": a.detector, "trackers": a.trackers,
              "params": params, "min_conf": a.min_conf, "seqs": seqs, "fps_assumed": 10.0}
    result = {"run_id": a.run_id, **run_metadata(config), "cpu": platform.processor() or platform.machine(),
              "scores": scores, "timing": timing}
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.rmtree(gt_root)  # only symlinks; GT stays in DATA

    print(f"{a.run_id}: {a.bench} {a.split} detector={a.detector} ({len(seqs)} seqs)")
    print(f"{'tracker':<10} {'HOTA':>6} {'DetA':>6} {'AssA':>6} {'IDF1':>6} {'MOTA':>7} {'IDSW':>6} {'ms/f':>7}")
    for name in a.trackers:
        r, t = scores[name], timing[name]
        print(f"{name:<10} {r['HOTA']:6.2f} {r['DetA']:6.2f} {r['AssA']:6.2f} {r['IDF1']:6.2f} "
              f"{r['MOTA']:7.2f} {int(r['IDSW']):6d} {t['track_ms_per_frame']:7.3f}")


if __name__ == "__main__":
    main()
