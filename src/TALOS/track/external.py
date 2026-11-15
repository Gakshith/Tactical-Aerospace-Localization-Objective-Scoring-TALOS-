"""Published baseline trackers (Roboflow ``trackers`` 2.7.0) behind the TALOS
``Tracker`` interface, so they run on exactly the same cached detections as
our own tracker.

Registered names: ``sort``, ``bytetrack``, ``ocsort``, ``botsort``.

    trk = REGISTRY["bytetrack"](fps=10.0)             # library defaults
    trk = REGISTRY["bytetrack"](fps=10.0, minimum_iou_threshold=0.05)

Contract (see base.py): ``update(dets (k,5) [x1,y1,x2,y2,conf])`` ->
``(m,6) [x1,y1,x2,y2,track_id,conf]``, called on every frame.

How the library is adapted, and why each choice is fair to the baseline:

* Input: ``sv.Detections(xyxy, confidence)``. No ``class_id``: none of the four
  trackers reads it (association is class-agnostic), and our protocol has none.
* Output: the library returns rows of the *input* detections with a
  ``tracker_id``; ``-1`` marks a detection with no confirmed track (tentative
  track, unmatched low-confidence det, or a det that just spawned a track).
  We drop those rows, so only confirmed tracks are reported. Boxes and conf
  are therefore the matched detection's own box and confidence (the library
  does not report Kalman boxes for coasting tracks; nor do we).
* IDs: the library allocates 0-based IDs per tracker instance; we add 1 so
  IDs are positive. The mapping is a constant shift, so stability is the
  library's own.
* ``frame_rate`` = the sequence fps. The library expresses
  ``lost_track_buffer`` in *30-fps frames* and rescales it by
  ``frame_rate / 30``, i.e. the default 30 means "1 second". At our ~10 fps
  that is 10 missed frames -- the library's intended semantics, kept as is.
* No ``timestamp`` is passed: fixed-rate mode (one Kalman step per call),
  matching how base.run_sequence feeds frames.
* Every library default is kept, except BoT-SORT's CMC (below). Overrides go
  through ``**params`` using the library's own keyword names.

Defaults worth knowing before reading results (kept, to be tuned on val with
the same budget as every other tracker):

* Confidence gates are calibrated for YOLOX on MOT17. ByteTrack/BoT-SORT only
  *start* tracks from dets with conf >= 0.7 (``track_activation_threshold``);
  OC-SORT ignores every det below 0.6 (``high_conf_det_threshold``) entirely;
  SORT starts tracks from conf >= 0.25. If our detector's scores run lower,
  these baselines will look worse for that reason alone.
* IoU gates (SORT 0.3, ByteTrack 0.1, OC-SORT 0.3, BoT-SORT 0.2 on IoU*conf
  and 0.5 for the low-conf pass) assume pedestrians tens to hundreds of px
  tall. A 10 px ship that moves 4 px/frame already has IoU ~0.43 with its
  previous box; a 5 px target moving 4 px has ~0.11. Any Kalman error on
  3-34 px objects fails these gates fast. That is a real weakness of IoU
  association on our data, not something the wrapper should paper over.
"""
from __future__ import annotations

import importlib.metadata
import inspect
from typing import Any

import numpy as np
import supervision as sv
import trackers  # ImportError here is what makes base.get() skip this module

from src.TALOS.track.base import register

LIBRARY_VERSION = importlib.metadata.version("trackers")  # the package has no __version__


class LibraryTracker:
    """Adapts one ``trackers`` instance to the TALOS ``Tracker`` protocol."""

    def __init__(self, name: str, impl: Any, params: dict[str, Any]):
        self.name = name
        self.impl = impl
        self.params = params  # fully resolved constructor kwargs, for logging

    def update(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 5)
        sv_dets = sv.Detections(
            xyxy=dets[:, :4].copy(),
            confidence=dets[:, 4].astype(np.float32),
        )
        res = self.impl.update(sv_dets)
        if len(res) == 0 or res.tracker_id is None:
            return np.zeros((0, 6))
        tid = np.asarray(res.tracker_id, dtype=np.int64)
        keep = tid >= 0  # -1 = no confirmed track for this detection
        if not keep.any():
            return np.zeros((0, 6))
        conf = (np.asarray(res.confidence, dtype=np.float64)[keep]
                if res.confidence is not None else np.ones(int(keep.sum())))
        return np.column_stack([np.asarray(res.xyxy, dtype=np.float64)[keep],
                                (tid[keep] + 1).astype(np.float64), conf])


def _resolved(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    out = {k: p.default for k, p in sig.parameters.items()
           if k != "self" and p.default is not inspect.Parameter.empty}
    unknown = set(kwargs) - set(out)
    if unknown:
        raise TypeError(f"{cls.__name__} has no parameter(s) {sorted(unknown)}")
    out.update(kwargs)
    return out


def _build(name: str, cls: type, fps: float, params: dict[str, Any],
           overrides: dict[str, Any] | None = None) -> LibraryTracker:
    if "frame_rate" in params:
        raise TypeError("pass the sequence rate as fps=, not frame_rate=")
    kwargs = {**(overrides or {}), **params, "frame_rate": float(fps)}
    resolved = _resolved(cls, kwargs)
    return LibraryTracker(name, cls(**kwargs), resolved)


@register("sort")
def make_sort(fps: float = 30.0, **params) -> LibraryTracker:
    return _build("sort", trackers.SORTTracker, fps, params)


@register("bytetrack")
def make_bytetrack(fps: float = 30.0, **params) -> LibraryTracker:
    return _build("bytetrack", trackers.ByteTrackTracker, fps, params)


@register("ocsort")
def make_ocsort(fps: float = 30.0, **params) -> LibraryTracker:
    # enable_cmc already defaults to False (the original OC-SORT); we never
    # have frames to give it, so refuse to turn it on.
    if params.get("enable_cmc"):
        raise ValueError("ocsort: CMC needs video frames; the TALOS interface passes boxes only")
    return _build("ocsort", trackers.OCSORTTracker, fps, params)


@register("botsort")
def make_botsort(fps: float = 30.0, **params) -> LibraryTracker:
    # Changed default: enable_cmc=False (library default True).
    # CMC estimates camera motion from the image, and our interface passes
    # boxes only. With enable_cmc=True and no frame, the library silently
    # skips CMC every step, so the result would be identical but the config
    # would claim CMC was on. AIR-MOT frames are already registered, so there
    # is little camera motion to compensate. Revisit by passing frames (or a
    # precomputed cmc_transform) if a benchmark has a moving camera.
    #
    # ReID stays off (reid_model=None, the library default): our objects are
    # 3-34 px wide, too small for an appearance embedding, and update() would
    # need the frame. This is the library's "geometry-only BoT-SORT baseline".
    if params.get("enable_cmc") or params.get("reid_model") is not None:
        raise ValueError("botsort: CMC and ReID need video frames; the TALOS interface passes boxes only")
    return _build("botsort", trackers.BoTSORTTracker, fps, params,
                  overrides={"enable_cmc": False})
