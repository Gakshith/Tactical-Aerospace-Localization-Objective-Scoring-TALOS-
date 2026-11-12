"""Evaluate threat warnings on real trajectories, with labels taken from the
future instead of invented.

For every moving ground-truth track, protected zones are placed:
  on-path    centred on where the object really is ``lead_s`` seconds after
             an anchor time, so it does enter (positives)
  near-miss  the same zone shifted sideways off the path by
             ``miss_offset`` radii, so it passes close but does not enter
Every other track within ``context_px`` of a zone is scored against it too
(parked aircraft beside a runway are the realistic false-alarm surface).

At each evaluation time t the warning model sees only the track's past: a
Kalman filter (TALOS-Track's) run over its boxes up to t, with optional
detector-like jitter. The label is whether the object's true future path
enters the zone in (t, t + H]. Times when the object is already inside are
skipped.

Metrics, for v2 (entry probability) and v1 (hand-weighted score):
  AUROC, average precision over all (track, zone, t) events;
  false alarms: the score threshold that flags ``fa_rate`` of negative
  events, and at that threshold the median warning lead time (entry time
  minus first alert) over positive zones, plus the share of entries warned;
  calibration of v2: reliability bins and expected calibration error.

    python -m src.TALOS.threat.evaluate --benches AIRMOT_plane AIRMOT_ship --splits train val
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.TALOS.data.mot import read_mot, read_seqinfo
from src.TALOS.data.split import load_split
from src.TALOS.threat.model import ThreatConfig, Zone, entry_probability, v1_score
from src.TALOS.track.talos_track import Config, TalosTracker

DATA = Path(os.environ.get("DATA", Path.home() / "talos" / "data"))


def tracks_of(gt: np.ndarray) -> dict[int, np.ndarray]:
    """id -> (n, 5) [frame, cx, cy, w, h] sorted by frame."""
    out = {}
    for tid in np.unique(gt[:, 1]).astype(int):
        r = gt[gt[:, 1] == tid]
        r = r[np.argsort(r[:, 0])]
        out[tid] = np.column_stack([r[:, 0], r[:, 2] + r[:, 4] / 2, r[:, 3] + r[:, 5] / 2, r[:, 4], r[:, 5]])
    return out


def filtered_states(track: np.ndarray, fps: float, jitter: float, rng) -> dict[int, tuple]:
    """Run TALOS-Track's filter over one track's boxes; frame -> (x, P)."""
    trk = TalosTracker(fps, Config(n_init=1, max_age_s=1e9, two_stage=False, high_conf=0.0))
    states = {}
    for f, cx, cy, w, h in track:
        cx_, cy_ = cx + rng.normal(0, jitter), cy + rng.normal(0, jitter)
        trk.update(np.array([[cx_ - w / 2, cy_ - h / 2, cx_ + w / 2, cy_ + h / 2, 1.0]]))
        t = trk.tracks[0]
        states[int(f)] = (t.x.copy(), t.P.copy())
    return states


def make_zones(track: np.ndarray, fps: float, radius: float, lead_s: float,
               miss_offset: float, n_anchors: int) -> list[tuple[str, Zone]]:
    zones = []
    lead = int(round(lead_s * fps))
    if len(track) <= lead + 2:
        return zones
    anchors = np.linspace(0, len(track) - lead - 1, n_anchors).astype(int)
    for a in np.unique(anchors):
        p_now, p_fut = track[a, 1:3], track[a + lead, 1:3]
        heading = p_fut - p_now
        n = np.linalg.norm(heading)
        if n < radius:  # barely moves over the lead time: not a meaningful approach
            continue
        side = np.array([-heading[1], heading[0]]) / n
        zones.append(("on-path", Zone.circle(*p_fut, radius)))
        zones.append(("near-miss", Zone.circle(*(p_fut + side * miss_offset * radius), radius)))
    return zones


def evaluate(benches, splits, cfg: ThreatConfig, radius=25.0, lead_s=8.0, miss_offset=2.5,
             n_anchors=3, min_speed=0.3, context_px=300.0, jitter=2.0, step=5,
             fa_rate=0.05, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    events = []   # (zone_key, is_positive_zone, frame, label, p_v2, s_v1, entry_frame)
    horizon = int(round(cfg.horizon_s * cfg.fps))
    n_zones = {"on-path": 0, "near-miss": 0}
    for bench in benches:
        for split in splits:
            for seq in load_split("airmot")[split]:
                sdir = DATA / "mot" / f"{bench}-all" / seq
                if not sdir.is_dir():
                    continue
                fps = read_seqinfo(sdir).fps
                tracks = tracks_of(read_mot(sdir / "gt" / "gt.txt"))
                movers = {k: t for k, t in tracks.items()
                          if len(t) > 2 and np.linalg.norm(t[-1, 1:3] - t[0, 1:3]) / len(t) >= min_speed}
                states = {}
                for zid, (tid, tr) in enumerate(movers.items()):
                    for kind, zone in make_zones(tr, fps, radius, lead_s, miss_offset, n_anchors):
                        n_zones[kind] += 1
                        zkey = f"{bench}/{seq}/{tid}/{zid}/{kind}/{n_zones[kind]}"
                        for oid, other in tracks.items():
                            near = np.min(np.linalg.norm(other[:, 1:3] - zone.centre, axis=1))
                            if oid != tid and near > context_px:
                                continue
                            if oid not in states:
                                states[oid] = filtered_states(other, fps, jitter, rng)
                            inside = zone.contains(other[:, 1:3])
                            frames = other[:, 0].astype(int)
                            entry = frames[inside][0] if inside.any() else None
                            for i in range(2, len(other), step):
                                f = frames[i]
                                if inside[i]:
                                    continue
                                fut = inside[(frames > f) & (frames <= f + horizon)]
                                label = bool(fut.any())
                                x, P = states[oid][f]
                                p, _ = entry_probability(x, P, zone, cfg)
                                events.append((zkey, oid == tid and kind == "on-path", int(f),
                                               label, p, v1_score(x, P, zone, fps=fps),
                                               entry if label else None))
    return summarise(events, n_zones, fa_rate, cfg)


def auroc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s):
        m = s == v
        ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def avg_precision(y, s):
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def lead_times(events, score_idx, thr, fps):
    """Per entered zone/track pair: seconds between first alert and entry."""
    first_alert, entry = {}, {}
    for e in events:
        key = (e[0],)
        if e[6] is not None:
            entry[key] = e[6]
        if e[score_idx] >= thr and e[3]:
            first_alert[key] = min(first_alert.get(key, 1e18), e[2])
    leads = [(entry[k] - first_alert[k]) / fps for k in entry if k in first_alert]
    return leads, len(entry)


def summarise(events, n_zones, fa_rate, cfg) -> dict:
    if not events:
        raise SystemExit("no events: no moving tracks in the chosen splits")
    y = np.array([e[3] for e in events])
    out = {"n_events": len(events), "n_positive_events": int(y.sum()), "zones": n_zones,
           "config": asdict(cfg)}
    for name, idx in (("v2_entry_prob", 4), ("v1_score", 5)):
        s = np.array([e[idx] for e in events], float)
        thr = float(np.quantile(s[~y], 1 - fa_rate)) if (~y).any() else 1.0
        thr = max(thr, 1e-9)
        leads, n_entered = lead_times(events, idx, thr, cfg.fps)
        out[name] = {
            "auroc": round(auroc(y, s), 4), "avg_precision": round(avg_precision(y, s), 4),
            f"threshold_at_{int(fa_rate*100)}pct_false_alarms": round(thr, 4),
            "entries_warned": f"{len(leads)}/{n_entered}",
            "median_lead_s": round(float(np.median(leads)), 2) if leads else None,
        }
    p = np.array([e[4] for e in events])
    bins = np.linspace(0, 1, 11)
    rel, ece = [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            rel.append({"bin": f"{lo:.1f}-{hi:.1f}", "mean_p": round(float(p[m].mean()), 3),
                        "observed": round(float(y[m].mean()), 3), "n": int(m.sum())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    brier = float(np.mean((p - y) ** 2))
    out["v2_calibration"] = {"ece": round(float(ece), 4), "brier": round(brier, 5), "reliability": rel}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", nargs="+", default=["AIRMOT_plane", "AIRMOT_ship"])
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--horizon-s", type=float, default=10.0)
    ap.add_argument("--accel-std", type=float, default=20.0)
    ap.add_argument("--radius", type=float, default=25.0)
    ap.add_argument("--lead-s", type=float, default=8.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if "test" in a.splits:
        Path("results").mkdir(exist_ok=True)
        with open("results/TEST_LOG.md", "a") as f:
            f.write(f"- threat eval on test: {a.benches}\n")
    cfg = ThreatConfig(horizon_s=a.horizon_s, accel_std=a.accel_std)
    res = evaluate(a.benches, a.splits, cfg, radius=a.radius, lead_s=a.lead_s)
    res.update(benches=a.benches, splits=a.splits, radius_px=a.radius, lead_s=a.lead_s,
               fps_assumed=10.0)
    text = json.dumps(res, indent=2)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
