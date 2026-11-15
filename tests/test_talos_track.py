import numpy as np
import pytest

from src.TALOS.track.base import get, run_sequence
from src.TALOS.track.talos_track import Config, TalosTracker, xyxy_to_z, z_to_xyxy


def box(cx, cy, s=8, conf=0.9):
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2, conf]


def frames_two_tiny_movers(n=40, drop=range(20, 24), fp_at=15):
    """Two 8 px objects at 4 px/frame (IoU between frames ~0.33) crossing paths."""
    out = []
    for f in range(n):
        dets = [box(100 + 4 * f, 100 + 2 * f)]
        if f not in drop:
            dets.append(box(260 - 4 * f, 140 - f))
        if f == fp_at:
            dets.append(box(900, 900, conf=0.95))
        out.append(np.array(dets, dtype=np.float32))
    return out


def ids_per_object(rows, n_frames):
    """Map each output row to the nearest true object; return ids seen per object."""
    seen = {0: set(), 1: set()}
    for r in rows:
        f = int(r[0]) - 1
        cx, cy = r[2] + r[4] / 2, r[3] + r[5] / 2
        a = np.hypot(cx - (100 + 4 * f), cy - (100 + 2 * f))
        b = np.hypot(cx - (260 - 4 * f), cy - (140 - f))
        if min(a, b) < 6:
            seen[0 if a < b else 1].add(int(r[1]))
    return seen


def test_box_state_roundtrip():
    b = np.array([[10, 20, 18, 30]], float)
    assert np.allclose(z_to_xyxy(xyxy_to_z(b)), b)


def test_full_tracker_keeps_ids_through_dropout_and_ignores_false_positive():
    res = run_sequence(TalosTracker(fps=10), frames_two_tiny_movers())
    seen = ids_per_object(res.rows, 40)
    assert len(seen[0]) == 1 and len(seen[1]) == 1, seen
    assert seen[0] != seen[1]
    # the one-frame false positive at (900, 900) never becomes a confirmed track
    far = res.rows[(res.rows[:, 2] > 800)]
    assert len(far) == 0


def test_tentative_tracks_hidden_until_n_init():
    trk = TalosTracker(fps=10, config=Config(n_init=3))
    outs = [trk.update(np.array([box(50 + 3 * f, 50)], np.float32)) for f in range(3)]
    assert [len(o) for o in outs] == [0, 0, 1]


def test_lost_tracks_are_deleted_after_max_age():
    trk = TalosTracker(fps=10, config=Config(max_age_s=0.5, n_init=1))
    trk.update(np.array([box(50, 50)], np.float32))
    for _ in range(6):
        trk.update(np.zeros((0, 5), np.float32))
    assert trk.tracks == []


def test_predict_runs_on_empty_frames_with_real_dt():
    trk = TalosTracker(fps=10, config=Config(n_init=1))
    for f in range(10):  # 4 px/frame = 40 px/s
        trk.update(np.array([box(100 + 4 * f, 100)], np.float32))
    trk.update(np.zeros((0, 5), np.float32))
    t = trk.tracks[0]
    assert t.x[2] == pytest.approx(40, rel=0.15)        # velocity in px/s
    assert t.x[0] == pytest.approx(100 + 4 * 10, abs=2)  # predicted one frame ahead


def test_low_confidence_extends_but_never_starts_tracks():
    trk = TalosTracker(fps=10, config=Config(n_init=1, high_conf=0.5, low_conf=0.1))
    trk.update(np.array([box(50, 50, conf=0.9)], np.float32))
    out = trk.update(np.array([box(53, 50, conf=0.2), box(400, 400, conf=0.3)], np.float32))
    assert len(out) == 1 and out[0, 4] == 1  # extended track 1, no new track for the weak box
    assert len(trk.tracks) == 1


def test_two_stage_off_drops_weak_boxes():
    trk = TalosTracker(fps=10, config=Config(n_init=1, two_stage=False))
    trk.update(np.array([box(50, 50, conf=0.9)], np.float32))
    out = trk.update(np.array([box(53, 50, conf=0.2)], np.float32))
    assert len(out) == 0


def test_reupdate_gives_sane_velocity_after_gap():
    cfg = Config(n_init=1, max_age_s=3)
    a, b = TalosTracker(10, cfg), TalosTracker(10, Config(n_init=1, max_age_s=3, reupdate=False))
    for trk in (a, b):
        for f in range(8):
            trk.update(np.array([box(100 + 4 * f, 100)], np.float32))
        for _ in range(6):   # object drifts sideways by 6 px while unseen
            trk.update(np.zeros((0, 5), np.float32))
        trk.update(np.array([box(128 + 4 * 7, 106)], np.float32))
    for trk in (a, b):
        assert len(trk.tracks) == 1, "the re-detection must extend the same track"
    # both variants must recover the true motion after the gap:
    # sideways drift ~6 px / 0.7 s ~ 8.6 px/s, forward speed still 40 px/s.
    # (Whether re-update helps on real data is decided by the ablation, not here:
    # covariance growth while coasting already lets a single update correct vy.)
    for trk in (a, b):
        assert trk.tracks[0].x[3] == pytest.approx(8.6, abs=2.5)
        assert trk.tracks[0].x[2] == pytest.approx(40, abs=8)


def test_registry_and_param_validation():
    make = get("talos")
    assert isinstance(make(fps=10), TalosTracker)
    with pytest.raises(TypeError):
        make(fps=10, not_a_param=1)


def test_calibration_recovers_injected_jitter():
    from src.TALOS.detect.cache import degrade
    from src.TALOS.track.calibrate import fit, residuals
    rows, gt = [], []
    for f in range(1, 101):
        for k in range(10):
            x, s = 60 * k + 10, 12 + 4 * k            # sizes 12..48 px
            rows.append([f, x, 50, x + s, 50 + s, 1.0])
            gt.append([f, k + 1, x, 50, s, s, 1, 1, 1])
    dets = degrade(np.array(rows, np.float32), (1920, 1080), jitter=2.0, seed=0)
    cal = fit(residuals(dets, np.array(gt, float)))
    # injected noise is a flat 2 px at every size; IoU>=0.5 matching trims tails slightly
    assert 1.6 < cal["meas_pos_min"] < 2.3
    assert all(1.6 < b["sigma_px"] < 2.4 for b in cal["bins"])


def test_dedup_stops_twin_tracks_on_one_object():
    """A detector that sometimes adds a shifted second box on the same object
    must not create a twin track that steals the ID back and forth."""
    def frames():
        out = []
        for f in range(30):
            dets = [box(100 + 2 * f, 100, s=30)]
            if f % 3 == 1:
                dets.append(box(104 + 2 * f, 103, s=30, conf=0.85))  # stray duplicate
            out.append(np.array(dets, np.float32))
        return out
    on = run_sequence(TalosTracker(10, Config(n_init=1)), frames())
    off = run_sequence(TalosTracker(10, Config(n_init=1, dedup=False)), frames())
    assert len(np.unique(on.rows[:, 1])) == 1
    assert len(np.unique(off.rows[:, 1])) > 1
