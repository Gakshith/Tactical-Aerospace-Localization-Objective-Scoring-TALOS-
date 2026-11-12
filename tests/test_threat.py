import numpy as np
import pytest

from src.TALOS.threat.model import ThreatConfig, Zone, entry_probability, v1_score
from src.TALOS.threat.evaluate import auroc, avg_precision


def test_zone_contains_circle():
    z = Zone.circle(0, 0, 10)
    pts = np.array([[0, 0], [9, 0], [11, 0], [0, -9.5]])
    assert z.contains(pts).tolist() == [True, True, False, True]


def _state(x, y, vx, vy, pos_var=4.0, vel_var=4.0):
    return np.array([x, y, vx, vy, 10, 10.0]), np.diag([pos_var, pos_var, vel_var, vel_var, 1, 1])


def test_heading_straight_at_zone_is_near_certain_and_tte_is_right():
    # 20 px/s towards a zone 100 px away, with a well-settled velocity estimate
    x, P = _state(0, 0, 20, 0, pos_var=1.0, vel_var=0.25)
    p, tte = entry_probability(x, P, Zone.circle(100, 0, 10), ThreatConfig(horizon_s=10, accel_std=1))
    assert p > 0.95
    assert tte == pytest.approx(4.5, abs=0.4)       # reaches the zone edge at 90 px


def test_heading_away_or_too_slow_is_near_zero():
    cfg = ThreatConfig(horizon_s=10, accel_std=1)
    assert entry_probability(*_state(0, 0, -20, 0), Zone.circle(100, 0, 10), cfg)[0] < 0.01
    assert entry_probability(*_state(0, 0, 2, 0), Zone.circle(100, 0, 10), cfg)[0] < 0.01


def test_uncertainty_spreads_probability():
    cfg = ThreatConfig(horizon_s=10, accel_std=1)
    zone = Zone.circle(100, 30, 10)                  # off the current heading
    sure = entry_probability(*_state(0, 0, 20, 0, vel_var=0.01), zone, cfg)[0]
    unsure = entry_probability(*_state(0, 0, 20, 0, vel_var=100), zone, cfg)[0]
    assert sure < 0.01 < unsure < 0.6


def test_deterministic_with_seed():
    x, P = _state(0, 0, 20, 3, vel_var=50)
    z = Zone.circle(100, 0, 15)
    assert entry_probability(x, P, z)[0] == entry_probability(x, P, z)[0]


def test_v1_score_prefers_approaching():
    z = Zone.circle(100, 0, 10)
    toward = v1_score(*_state(0, 0, 20, 0), z)
    away = v1_score(*_state(0, 0, -20, 0), z)
    assert toward > away


def test_metric_helpers():
    y = np.array([0, 0, 1, 1], bool)
    assert auroc(y, np.array([.1, .2, .8, .9])) == 1.0
    assert auroc(y, np.array([.5, .5, .5, .5])) == 0.5
    assert avg_precision(y, np.array([.1, .2, .8, .9])) == 1.0
