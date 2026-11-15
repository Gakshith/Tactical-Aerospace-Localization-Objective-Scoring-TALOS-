"""Equal-budget hyperparameter search for every tracker, on val only.

A comparison is fair only if each tracker gets the same chance to fit the
data. Every tracker gets ``--budget`` configurations: its library defaults
first, then seeded random draws from the search space below. All configs of
all trackers run on the same cached detections, and the best by HOTA on the
val split is kept. Test is never touched here; the chosen configs are run on
test once, later, with ``bench``.

    python -m src.TALOS.eval.tune --bench AIRMOT_plane --detector y26m_t640 \
        --trackers sort bytetrack ocsort botsort talos --budget 30 \
        --fixed '{"talos": {"meas_pos_min": 1.8, "meas_pos_frac": 0.044}}' --run-id tune-...

Search spaces follow each library's own tuning ranges where it publishes
them (the trackers library's grids go down to IoU 0.05 for SORT/ByteTrack/
BoT-SORT, 0.1 for OC-SORT); thresholds span the detector's score range;
lost-track buffers span 0.5-4 s.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.TALOS.eval.bench import DATA, evaluate_trackers
from src.TALOS.eval.trackeval_runner import run_metadata

LOST = [5, 10, 20, 30, 60, 120]          # library units: frames at 30 fps (rescaled to fps)
SPACES: dict[str, dict[str, list]] = {
    "sort": {
        "track_activation_threshold": [0.1, 0.2, 0.25, 0.3, 0.4, 0.5],
        "minimum_iou_threshold": [0.05, 0.1, 0.2, 0.3],
        "minimum_consecutive_frames": [1, 2, 3, 5],
        "lost_track_buffer": LOST,
    },
    "bytetrack": {
        "track_activation_threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "high_conf_det_threshold": [0.2, 0.3, 0.4, 0.5, 0.6],
        "minimum_iou_threshold": [0.05, 0.1, 0.2, 0.3],
        "minimum_consecutive_frames": [1, 2, 3, 5],
        "lost_track_buffer": LOST,
    },
    "ocsort": {
        "high_conf_det_threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "minimum_iou_threshold": [0.1, 0.2, 0.3],
        "minimum_consecutive_frames": [1, 2, 3, 5],
        "direction_consistency_weight": [0.0, 0.1, 0.2, 0.5],
        "delta_t": [1, 3, 5],
        "lost_track_buffer": LOST,
    },
    "botsort": {
        "track_activation_threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "high_conf_det_threshold": [0.2, 0.3, 0.4, 0.5, 0.6],
        "minimum_iou_threshold_first_assoc": [0.05, 0.1, 0.2, 0.3],
        "minimum_iou_threshold_second_assoc": [0.05, 0.1, 0.3, 0.5],
        "minimum_consecutive_frames": [1, 2, 3, 5],
        "lost_track_buffer": LOST,
    },
    "talos": {
        "high_conf": [0.2, 0.3, 0.4, 0.5, 0.6],
        "low_conf": [0.05, 0.1, 0.15],
        "n_init": [1, 2, 3, 5],
        "max_age_s": [0.5, 1.0, 2.0, 4.0],
        "gate_p": [0.95, 0.99, 0.999],
        "accel_std": [5.0, 10.0, 20.0, 40.0],
        "coast_output": [False, True],
        "coast_frames": [1, 3, 5],
    },
}


def draw(space: dict[str, list], rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in space.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--trackers", nargs="+", required=True)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective", default="HOTA")
    ap.add_argument("--fixed", default="{}", help="JSON {tracker: {param: value}} held fixed")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-root", type=Path, default=Path("results"))
    a = ap.parse_args()

    out = a.out_root / a.run_id
    if out.exists():
        raise SystemExit(f"{out} exists; run ids are never reused")
    fixed = json.loads(a.fixed)
    configs: dict[str, dict] = {}
    for t in a.trackers:
        rng = random.Random(f"{a.seed}-{t}")  # per-tracker stream: adding a tracker changes nothing else
        seen = set()
        cands = [{}]  # library defaults always count as one of the budget
        while len(cands) < a.budget:
            c = draw(SPACES[t], rng)
            key = json.dumps(c, sort_keys=True)
            if key not in seen:
                seen.add(key)
                cands.append(c)
        for i, c in enumerate(cands):
            configs[f"{t}:c{i:02d}"] = {**fixed.get(t, {}), **c}

    names = list(configs)
    scores, timing, seqs = evaluate_trackers(a.bench, "val", a.detector, names, configs, out)

    best = {}
    for t in a.trackers:
        mine = [n for n in names if n.split(":")[0] == t]
        b = max(mine, key=lambda n: scores[n][a.objective])
        best[t] = {"config_name": b, "params": configs[b], "scores": scores[b],
                   "default_scores": scores[f"{t}:c00"]}
    result = {
        "run_id": a.run_id,
        **run_metadata({"bench": a.bench, "detector": a.detector, "split": "val", "budget": a.budget,
                        "seed": a.seed, "objective": a.objective, "fixed": fixed, "seqs": seqs}),
        "best": best,
        "all": {n: {"params": configs[n], "scores": scores[n], "timing": timing[n]} for n in names},
    }
    (out / "tuning.json").write_text(json.dumps(result, indent=2) + "\n")
    import shutil
    shutil.rmtree(out / "trk")  # hundreds of MOT files; the scores are what we keep

    print(f"{a.run_id}: {a.bench} val, detector={a.detector}, budget {a.budget}/tracker, "
          f"objective {a.objective}")
    print(f"{'tracker':<10} {'default':>8} {'tuned':>8} {'IDF1':>7} {'MOTA':>7} {'IDSW':>6}")
    for t, b in best.items():
        s = b["scores"]
        print(f"{t:<10} {b['default_scores'][a.objective]:8.2f} {s[a.objective]:8.2f} "
              f"{s['IDF1']:7.2f} {s['MOTA']:7.2f} {int(s['IDSW']):6d}")


if __name__ == "__main__":
    main()
