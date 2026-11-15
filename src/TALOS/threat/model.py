"""Threat model v2: probability that a track enters a protected zone soon.

v1 ranked tracks with a hand-weighted score (0.4 urgency + 0.3 proximity +
0.2 intent + 0.1 certainty) in pixels, against the frame's top-left corner.
v2 answers one question in physical units, with the tracker's own
uncertainty:

    P( track enters zone Z within the next H seconds )

The track's current Kalman state (position, velocity) and covariance are
sampled; each sample is propagated forward under the same constant-velocity
plus white-acceleration model the tracker uses, and the probability is the
fraction of sampled futures that cross into the zone. The zone is any
polygon, in pixels; ``gsd_m`` converts pixels to metres for reporting.

Monte Carlo keeps this exact for arbitrary polygons and non-Gaussian
crossing times (a closed form exists only for simple zones). With a fixed
seed it is deterministic; 512 samples x H*fps steps is a few hundred
microseconds per track.

``v1_score`` re-implements v1's formula against the zone centre, so both can
be evaluated on the same events.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Zone:
    polygon: np.ndarray  # (k, 2) pixel vertices, in order

    @staticmethod
    def circle(cx: float, cy: float, r: float, n: int = 32) -> "Zone":
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return Zone(np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)]))

    @property
    def centre(self) -> np.ndarray:
        return self.polygon.mean(axis=0)

    def contains(self, pts: np.ndarray) -> np.ndarray:
        """Vectorised even-odd point-in-polygon for (..., 2) points."""
        x, y = pts[..., 0], pts[..., 1]
        poly = self.polygon
        inside = np.zeros(x.shape, dtype=bool)
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            crosses = (yi > y) != (yj > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = (xj - xi) * (y - yi) / (yj - yi) + xi
            inside ^= crosses & (x < xint)
            j = i
        return inside


@dataclass(frozen=True)
class ThreatConfig:
    horizon_s: float = 10.0
    fps: float = 10.0
    # px/s^2 white-acceleration noise. 1.0 was fitted on AIR-MOT train trajectories by
    # Brier score (grid 1-80; flat between 1 and 5) and confirmed on val: Brier 0.0419 vs
    # 0.0456 and AUROC 0.855 vs 0.813 against the earlier default of 20.
    # See results/threat/fit/ and results/threat/d4-threat-val-*.
    accel_std: float = 1.0
    n_samples: int = 512
    seed: int = 0


def entry_probability(x: np.ndarray, P: np.ndarray, zone: Zone,
                      cfg: ThreatConfig = ThreatConfig()) -> tuple[float, float]:
    """x: state [cx, cy, vx, vy, ...] (px, px/s); P: its covariance (>= 4x4).
    Returns (P(enter within horizon), median time-to-entry in s among the
    samples that enter, or inf)."""
    rng = np.random.default_rng(cfg.seed)
    mean, cov = np.asarray(x[:4], float), np.asarray(P, float)[:4, :4]
    cov = (cov + cov.T) / 2 + np.eye(4) * 1e-9
    s = rng.multivariate_normal(mean, cov, size=cfg.n_samples)       # (n, 4)
    dt = 1.0 / cfg.fps
    steps = int(round(cfg.horizon_s * cfg.fps))
    # all futures at once: acceleration noise integrated into velocity and position
    acc = rng.normal(0, cfg.accel_std, size=(steps, cfg.n_samples, 2))
    vel = s[None, :, 2:] + np.cumsum(acc, axis=0) * dt                 # v_k after step k
    v_prev = np.concatenate([s[None, :, 2:], vel[:-1]], axis=0)
    pos = s[None, :, :2] + np.cumsum(v_prev * dt + 0.5 * acc * dt * dt, axis=0)
    traj = np.concatenate([s[None, :, :2], pos], axis=0)                # (steps+1, n, 2)
    lo, hi = zone.polygon.min(axis=0), zone.polygon.max(axis=0)
    near = ((traj >= lo) & (traj <= hi)).all(axis=-1)                   # cheap bounding-box test
    inside = np.zeros(near.shape, bool)
    if near.any():
        inside[near] = zone.contains(traj[near])
    hit_any = inside.any(axis=0)
    first = np.where(hit_any, inside.argmax(axis=0) * dt, np.inf)
    entered = np.isfinite(first)
    p = float(entered.mean())
    tte = float(np.median(first[entered])) if entered.any() else float("inf")
    return p, tte


def v1_score(x: np.ndarray, P: np.ndarray, zone: Zone, tau: float = 5.0,
             prox_scale: float = 800.0, trace_scale: float = 100.0,
             fps: float = 10.0) -> float:
    """v1's risk score (model_run.ThreatAssessmentEngine) with the zone centre
    as origin. v1 measured velocity in px/frame, so it is converted back."""
    pos, vel = np.asarray(x[:2], float), np.asarray(x[2:4], float) / fps
    vec = zone.centre - pos
    dist = np.linalg.norm(vec)
    unit = vec / (dist + 1e-6)
    v_close = float(vel @ unit)
    urgency = np.exp(-(dist / v_close) / tau) if v_close > 0.5 else 0.0
    intent = float(np.clip(vel @ unit / (np.linalg.norm(vel) + 1e-6), 0, 1)) if v_close > 0 else 0.0
    proximity = np.exp(-dist / prox_scale)
    certainty = np.exp(-np.trace(np.asarray(P)[:2, :2]) / trace_scale)
    return float(0.4 * urgency + 0.3 * proximity + 0.2 * intent + 0.1 * certainty)
