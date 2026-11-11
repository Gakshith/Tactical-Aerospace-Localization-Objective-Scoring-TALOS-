"""TALOS-Track: a Kalman tracker built for tiny objects in satellite video.

Each feature that differs from v1 sits behind a flag, so the ablation table
is produced by switching flags, never by maintaining parallel code.

  lifecycle     tracks are tentative until ``n_init`` hits, coast while lost
                for up to ``max_age_s`` seconds, then are deleted. v1 kept
                every track forever, which turned each false positive into a
                permanent ghost.
  real_dt       dt = 1/fps of the video, and predict runs on every frame,
                including frames with no detections. v1 used dt = 1/30 and
                skipped predict on empty frames.
  maha_gate     association cost is the Mahalanobis distance of the
                innovation (centre and size) under the filter's own
                uncertainty S = H P H^T + R, gated at a chi-square quantile.
                IoU is useless for 5-10 px objects that move about their own
                size per frame, and a fixed pixel gate ignores how sure the
                filter is.
  two_stage     ByteTrack's idea: match confident detections first, then let
                low-confidence ones extend tracks that are still unmatched.
                Tiny objects often score low. New tracks start only from
                confident detections.
  reupdate      OC-SORT's observation-centric re-update: when a lost track is
                found again, rewind to its last real observation and re-run the
                filter through linearly interpolated virtual observations, so
                the velocity that accumulated error while coasting is replaced
                by one fitted to real data.
  coast_output  report a lost track's predicted box for up to
                ``coast_frames`` frames, filling short detector misses.

State is x = [cx, cy, vx, vy, w, h] (pixels, pixels/s) with a constant-
velocity model for the centre and a random walk for the size.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.TALOS.track.base import register

TENTATIVE, CONFIRMED, LOST = 0, 1, 2
CHI2_4DOF = {0.95: 9.488, 0.99: 13.277, 0.999: 18.467}
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]], dtype=float)
BIG = 1e9


@dataclass
class Config:
    # feature flags (all on = TALOS-Track; all off ~ v1 behaviour)
    lifecycle: bool = True
    real_dt: bool = True
    maha_gate: bool = True
    two_stage: bool = True
    reupdate: bool = True
    coast_output: bool = False
    # lifecycle
    n_init: int = 3
    max_age_s: float = 2.0
    # association
    gate_p: float = 0.99
    fixed_gate_px: float = 150.0      # used only when maha_gate is off (v1's rule)
    high_conf: float = 0.5
    low_conf: float = 0.1
    # noise model
    accel_std: float = 20.0           # px/s^2, white-acceleration process noise
    size_std: float = 1.0             # px per sqrt(s), size random walk
    meas_pos_frac: float = 0.05       # measurement std as a fraction of box size...
    meas_pos_min: float = 1.0         # ...but never below this many px
    init_vel_std: float = 50.0        # px/s, unknown initial velocity
    coast_frames: int = 3


@dataclass
class Track:
    id: int
    x: np.ndarray
    P: np.ndarray
    state: int = TENTATIVE
    hits: int = 1
    age_since_update: int = 0
    conf: float = 1.0
    last_box: np.ndarray = field(default_factory=lambda: np.zeros(4))
    # snapshot at the last real observation, for re-update
    obs_x: np.ndarray | None = None
    obs_P: np.ndarray | None = None
    obs_z: np.ndarray | None = None


def xyxy_to_z(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    return np.column_stack([b[:, 0] + w / 2, b[:, 1] + h / 2, w, h])


def z_to_xyxy(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float).reshape(-1, 4)
    return np.column_stack([z[:, 0] - z[:, 2] / 2, z[:, 1] - z[:, 3] / 2,
                            z[:, 0] + z[:, 2] / 2, z[:, 1] + z[:, 3] / 2])


class TalosTracker:
    def __init__(self, fps: float = 10.0, config: Config | None = None):
        self.cfg = config or Config()
        self.dt = 1.0 / fps if self.cfg.real_dt else 1.0 / 30
        self.max_age = max(1, round(self.cfg.max_age_s * fps))
        self.tracks: list[Track] = []
        self.next_id = 1
        self.F, self.Q = self._motion(self.dt)
        self.gate = CHI2_4DOF[self.cfg.gate_p]

    # -- Kalman pieces -----------------------------------------------------
    def _motion(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(6)
        F[0, 2] = F[1, 3] = dt
        q = self.cfg.accel_std ** 2
        Q = np.zeros((6, 6))
        for p, v in ((0, 2), (1, 3)):  # discrete white-noise acceleration
            Q[p, p] = dt ** 4 / 4 * q
            Q[p, v] = Q[v, p] = dt ** 3 / 2 * q
            Q[v, v] = dt ** 2 * q
        Q[4, 4] = Q[5, 5] = self.cfg.size_std ** 2 * dt
        return F, Q

    def _R(self, z: np.ndarray) -> np.ndarray:
        s = max(self.cfg.meas_pos_min, self.cfg.meas_pos_frac * max(z[2], z[3]))
        return np.diag([s * s, s * s, s * s, s * s])

    def _predict(self, t: Track) -> None:
        t.x = self.F @ t.x
        t.P = self.F @ t.P @ self.F.T + self.Q
        t.x[4:] = np.maximum(t.x[4:], 1.0)

    def _update(self, t: Track, z: np.ndarray) -> None:
        S = H @ t.P @ H.T + self._R(z)
        K = t.P @ H.T @ np.linalg.inv(S)
        t.x = t.x + K @ (z - H @ t.x)
        t.P = (np.eye(6) - K @ H) @ t.P

    def _new_track(self, box: np.ndarray, conf: float) -> None:
        z = xyxy_to_z(box)[0]
        x = np.array([z[0], z[1], 0.0, 0.0, z[2], z[3]])
        r = self._R(z)
        P = np.diag([r[0, 0], r[1, 1], self.cfg.init_vel_std ** 2, self.cfg.init_vel_std ** 2,
                     r[2, 2], r[3, 3]])
        t = Track(self.next_id, x, P, conf=conf, last_box=np.asarray(box, float))
        t.state = TENTATIVE if self.cfg.lifecycle and self.cfg.n_init > 1 else CONFIRMED
        t.obs_x, t.obs_P, t.obs_z = x.copy(), P.copy(), z.copy()
        self.tracks.append(t)
        self.next_id += 1

    # -- association -------------------------------------------------------
    def _cost(self, tracks: list[Track], boxes: np.ndarray) -> np.ndarray:
        if not tracks or not len(boxes):
            return np.zeros((len(tracks), len(boxes)))
        Z = xyxy_to_z(boxes)
        C = np.full((len(tracks), len(boxes)), BIG)
        for i, t in enumerate(tracks):
            hx = H @ t.x
            if self.cfg.maha_gate:
                # S depends on the detection only through R; use the track's
                # current size so one inverse serves the whole row
                Sinv = np.linalg.inv(H @ t.P @ H.T + self._R(hx))
                d = Z - hx
                m = np.einsum("ij,jk,ik->i", d, Sinv, d)
                ok = m <= self.gate
                C[i, ok] = m[ok]
            else:
                d = np.linalg.norm(Z - hx, axis=1)  # v1: distance over (cx, cy, w, h)
                ok = d < self.cfg.fixed_gate_px
                C[i, ok] = d[ok]
        return C

    @staticmethod
    def _match(C: np.ndarray) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        if C.size == 0:
            return [], set(range(C.shape[0])), set(range(C.shape[1]))
        r, c = linear_sum_assignment(C)
        pairs = [(i, j) for i, j in zip(r, c) if C[i, j] < BIG]
        mt = {i for i, _ in pairs}
        md = {j for _, j in pairs}
        return pairs, set(range(C.shape[0])) - mt, set(range(C.shape[1])) - md

    def _reupdate(self, t: Track, z_new: np.ndarray) -> None:
        """Rewind to the last real observation and replay the gap with
        interpolated virtual observations (OC-SORT's ORU)."""
        gap = t.age_since_update
        if not self.cfg.reupdate or gap <= 1 or t.obs_x is None:
            self._update(t, z_new)
            return
        t.x, t.P = t.obs_x.copy(), t.obs_P.copy()
        for k in range(1, gap + 1):
            self._predict(t)
            self._update(t, t.obs_z + (z_new - t.obs_z) * k / gap)

    def _apply(self, t: Track, box: np.ndarray, conf: float) -> None:
        z = xyxy_to_z(box)[0]
        self._reupdate(t, z)
        t.hits += 1
        t.age_since_update = 0
        t.conf = conf
        t.last_box = np.asarray(box, float)
        t.obs_x, t.obs_P, t.obs_z = t.x.copy(), t.P.copy(), z.copy()
        if t.state == TENTATIVE and t.hits >= self.cfg.n_init:
            t.state = CONFIRMED
        elif t.state == LOST:
            t.state = CONFIRMED

    # -- main loop ---------------------------------------------------------
    def update(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=float).reshape(-1, 5)
        cfg = self.cfg
        if cfg.real_dt or len(dets):  # v1 skipped predict on empty frames
            for t in self.tracks:
                self._predict(t)
                t.age_since_update += 1

        if cfg.two_stage:
            hi = dets[dets[:, 4] >= cfg.high_conf]
            lo = dets[(dets[:, 4] >= cfg.low_conf) & (dets[:, 4] < cfg.high_conf)]
        else:
            hi = dets[dets[:, 4] >= cfg.high_conf]
            lo = np.zeros((0, 5))

        # stage 1: confident detections against all tracks
        pairs, un_t, un_d = self._match(self._cost(self.tracks, hi[:, :4]))
        for i, j in pairs:
            self._apply(self.tracks[i], hi[j, :4], hi[j, 4])

        # stage 2: weak detections against confirmed/lost tracks left over
        if len(lo):
            rest = [self.tracks[i] for i in sorted(un_t) if self.tracks[i].state != TENTATIVE]
            pairs2, _, _ = self._match(self._cost(rest, lo[:, :4]))
            for i, j in pairs2:
                self._apply(rest[i], lo[j, :4], lo[j, 4])

        for j in sorted(un_d):
            self._new_track(hi[j, :4], hi[j, 4])

        # lifecycle
        alive = []
        for t in self.tracks:
            if t.age_since_update > 0:
                if not cfg.lifecycle:
                    alive.append(t)
                    continue
                if t.state == TENTATIVE:
                    continue  # a tentative track that misses once is dropped
                t.state = LOST
                if t.age_since_update > self.max_age:
                    continue
            alive.append(t)
        self.tracks = alive

        return self._output()

    def _output(self) -> np.ndarray:
        out = []
        for t in self.tracks:
            if t.state == TENTATIVE and self.cfg.lifecycle:
                continue
            if t.age_since_update == 0:
                box = t.last_box
            elif not self.cfg.lifecycle:
                box = t.last_box if not self.cfg.real_dt else z_to_xyxy(H @ t.x)[0]
            elif self.cfg.coast_output and t.age_since_update <= self.cfg.coast_frames:
                box = z_to_xyxy(H @ t.x)[0]
            else:
                continue
            out.append([*box, t.id, t.conf])
        return np.array(out, dtype=float).reshape(-1, 6)


@register("talos")
def make_talos(fps: float = 10.0, **params) -> TalosTracker:
    unknown = set(params) - set(Config.__dataclass_fields__)
    if unknown:
        raise TypeError(f"unknown TALOS-Track params: {sorted(unknown)}")
    return TalosTracker(fps=fps, config=Config(**params))
