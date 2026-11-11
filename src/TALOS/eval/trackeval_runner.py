"""Run the official TrackEval (HOTA / CLEAR / Identity) on MOT-format files.

Layout this module builds (TrackEval's MotChallenge2DBox convention)::

    <gt_root>/<BENCH>-<split>/<seq>/gt/gt.txt
    <gt_root>/<BENCH>-<split>/<seq>/seqinfo.ini
    <gt_root>/seqmaps/<BENCH>-<split>.txt
    <trk_root>/<BENCH>-<split>/<tracker>/data/<seq>.txt

TrackEval only evaluates its "pedestrian" class (id 1), so one object class
is evaluated per benchmark: converters write class-specific GT with class 1
(e.g. ``AIRMOT_plane``), and tracker output is written the same way.
"""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

HEADLINE = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "IDs", "GT_IDs", "FP", "FN")


def write_seqmap(gt_root: Path, bench: str, split: str, seqs: list[str]) -> Path:
    path = Path(gt_root) / "seqmaps" / f"{bench}-{split}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("name\n" + "".join(f"{s}\n" for s in seqs))
    return path


def run_trackeval(
    gt_root: Path,
    trk_root: Path,
    bench: str,
    split: str,
    trackers: list[str],
    seqs: list[str],
    out_dir: Path | None = None,
    num_cores: int = 8,
) -> dict[str, dict[str, float]]:
    """Evaluate ``trackers`` on ``seqs``; returns {tracker: {metric: value}}
    for the combined sequence set, with HOTA-family metrics as percentages."""
    import numpy as np

    # TrackEval (last release 2021) still uses aliases removed in NumPy 1.24.
    for alias, typ in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, alias):
            setattr(np, alias, typ)
    import trackeval  # imported lazily so the rest of TALOS runs without it

    seqmap = write_seqmap(gt_root, bench, split, seqs)
    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update(
        USE_PARALLEL=num_cores > 1,
        NUM_PARALLEL_CORES=num_cores,
        PRINT_RESULTS=False,
        PRINT_CONFIG=False,
        TIME_PROGRESS=False,
        OUTPUT_SUMMARY=out_dir is not None,
        OUTPUT_DETAILED=out_dir is not None,
        PLOT_CURVES=False,
    )
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(
        GT_FOLDER=str(gt_root),
        TRACKERS_FOLDER=str(trk_root),
        OUTPUT_FOLDER=str(out_dir) if out_dir else None,
        BENCHMARK=bench,
        SPLIT_TO_EVAL=split,
        TRACKERS_TO_EVAL=trackers,
        CLASSES_TO_EVAL=["pedestrian"],
        SEQMAP_FILE=str(seqmap),
        DO_PREPROC=False,  # MOT17 distractor removal does not apply to satellite data
        SKIP_SPLIT_FOL=False,
        PRINT_CONFIG=False,
    )
    evaluator = trackeval.Evaluator(eval_cfg)
    datasets = [trackeval.datasets.MotChallenge2DBox(ds_cfg)]
    quiet = {"PRINT_CONFIG": False}
    # Evaluator appends the Count metric itself; adding it here is an error.
    metrics = [trackeval.metrics.HOTA(quiet), trackeval.metrics.CLEAR(quiet),
               trackeval.metrics.Identity(quiet)]
    raw, _ = evaluator.evaluate(datasets, metrics)
    res = raw["MotChallenge2DBox"]
    return {t: _summarise(res[t]["COMBINED_SEQ"]["pedestrian"]) for t in trackers}


def _summarise(r: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for fam in ("HOTA", "CLEAR", "Identity", "Count"):
        for k, v in r.get(fam, {}).items():
            if k in HEADLINE:
                # HOTA-family values are arrays over localisation thresholds
                val = float(v.mean()) if hasattr(v, "mean") else float(v)
                out[k] = round(val * 100, 3) if k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA") else val
    return out


def run_metadata(config: dict) -> dict:
    """Provenance stamped into every metrics.json: what produced this number."""
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    cfg_bytes = json.dumps(config, sort_keys=True).encode()
    import hashlib
    sha, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    stamp = Path(".talos_source")  # written by scripts/sync_polaris.sh where .git is absent
    if sha == "unknown" and stamp.exists():
        kv = dict(line.split("=", 1) for line in stamp.read_text().split())
        sha, dirty = kv.get("git_sha", "unknown"), kv.get("git_dirty_files", "1") != "0"
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "config_hash": hashlib.sha256(cfg_bytes).hexdigest()[:12],
        "config": config,
        "host": platform.node(),
    }
