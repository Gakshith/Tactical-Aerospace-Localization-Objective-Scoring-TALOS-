"""The one interface every tracker in TALOS implements, so trackers can be
swapped in benchmarks and in the app without touching anything else.

A tracker is built once per sequence and fed one frame at a time:

    trk = make_tracker(fps=10.0, **params)
    for dets in frames:                 # dets: (k, 5) [x1, y1, x2, y2, conf]
        out = trk.update(dets)          # out:  (m, 6) [x1, y1, x2, y2, track_id, conf]

``update`` must be called for every frame, including frames with no
detections (pass a (0, 5) array), so trackers can predict through gaps.
Returned IDs are positive integers that are stable for the life of the track.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from src.TALOS.detect import cache as det_cache


class Tracker(Protocol):
    def update(self, dets: np.ndarray) -> np.ndarray: ...


TrackerFactory = Callable[..., Tracker]
REGISTRY: dict[str, TrackerFactory] = {}


def register(name: str):
    def deco(factory: TrackerFactory) -> TrackerFactory:
        if name in REGISTRY:
            raise ValueError(f"tracker {name!r} registered twice")
        REGISTRY[name] = factory
        return factory
    return deco


def get(name: str) -> TrackerFactory:
    # importing the modules fills REGISTRY; kept lazy so optional deps stay optional
    from src.TALOS.track import talos_track, v1  # noqa: F401
    try:
        from src.TALOS.track import external  # noqa: F401
    except ModuleNotFoundError as e:
        # only a missing optional library may silence this; a bug inside
        # external.py must surface rather than quietly drop the baselines
        if e.name not in ("trackers", "supervision"):
            raise
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown tracker {name!r}; known: {sorted(REGISTRY)}") from None


@dataclass
class SeqResult:
    rows: np.ndarray            # MOT rows: frame, id, x, y, w, h, conf, 1, 1
    n_frames: int
    seconds: float              # wall time spent inside update()
    per_frame_ms: list[float] = field(default_factory=list)


def run_sequence(tracker: Tracker, frames: list[np.ndarray]) -> SeqResult:
    """Feed per-frame detections through a tracker; return MOT rows."""
    rows, times = [], []
    for f, dets in enumerate(frames, start=1):
        t0 = time.perf_counter()
        out = np.asarray(tracker.update(np.asarray(dets, dtype=np.float32).reshape(-1, 5)), dtype=float)
        times.append((time.perf_counter() - t0) * 1000)
        if out.size:
            out = out.reshape(-1, 6)
            x1, y1, x2, y2, tid, conf = out.T
            rows.append(np.column_stack([np.full(len(out), f), tid, x1, y1, x2 - x1, y2 - y1,
                                         conf, np.ones(len(out)), np.ones(len(out))]))
    mot = np.vstack(rows) if rows else np.zeros((0, 9))
    if len(mot) and (mot[:, 1] <= 0).any():
        raise ValueError("tracker returned a non-positive track id")
    return SeqResult(mot, len(frames), sum(times) / 1000, times)


def frames_from_cache(root, detector: str, bench: str, seq: str, n_frames: int,
                      min_conf: float = 0.0) -> list[np.ndarray]:
    dets = det_cache.load(root, detector, bench, seq)
    if min_conf > 0 and len(dets):
        dets = dets[dets[:, 5] >= min_conf]
    return det_cache.by_frame(dets, n_frames)
