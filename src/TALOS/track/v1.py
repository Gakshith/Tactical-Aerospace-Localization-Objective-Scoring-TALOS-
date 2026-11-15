"""The original (v1) TALOS tracker behind the common interface: the "before"
baseline.

This wraps ``ModelRun.MultiObjectTracker`` from ``components/model_run.py``
unchanged, so the benchmark measures what v1 actually did, bugs included:

- no detections in a frame -> ``update`` returns early, so no predict step;
- tracks are never deleted, and every track is reported on every frame,
  so a lost track leaves a stale box behind;
- ``dt`` is fixed at 1/30 s regardless of the video's frame rate;
- a match is accepted only if its distance cost is below 150 px.

v1 dropped detections below confidence 0.5 before tracking; ``min_conf``
reproduces that.
"""
from __future__ import annotations

import numpy as np

from src.TALOS.track.base import register


class V1Tracker:
    def __init__(self, min_conf: float = 0.5, device: str = "cpu"):
        from src.TALOS.components.model_run import ModelRun  # heavy deps, import lazily

        self.min_conf = min_conf
        self.mot = ModelRun.MultiObjectTracker(dt=1 / 30, device=device)

    def update(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=float).reshape(-1, 5)
        keep = dets[dets[:, 4] >= self.min_conf]
        self.mot.update(keep[:, :4].tolist())
        out = []
        for t in self.mot.trackers:
            tid, x1, y1, x2, y2 = t.get_bounding_box()
            out.append([x1, y1, x2, y2, tid + 1, 1.0])  # v1 ids start at 0
        return np.array(out, dtype=float).reshape(-1, 6)


@register("v1")
def make_v1(fps: float = 30.0, **params) -> V1Tracker:
    # fps is ignored on purpose: v1 hard-codes dt = 1/30
    return V1Tracker(**params)
