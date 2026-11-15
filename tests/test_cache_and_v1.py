import numpy as np
import pytest

from src.TALOS.detect import cache
from src.TALOS.track.base import run_sequence


def test_cache_roundtrip_and_by_frame(tmp_path):
    d = np.array([[3, 0, 0, 5, 5, .9], [1, 1, 1, 4, 4, .8], [3, 9, 9, 12, 12, .7]])
    cache.save(tmp_path, "det", "B", "s", d)
    back = cache.load(tmp_path, "det", "B", "s")
    frames = cache.by_frame(back, 4)
    assert [len(f) for f in frames] == [1, 0, 2, 0]
    assert frames[0].shape == (1, 5) and frames[2][:, 4].tolist() == pytest.approx([.9, .7])


def test_cache_rejects_zero_based(tmp_path):
    with pytest.raises(ValueError):
        cache.save(tmp_path, "d", "B", "s", np.array([[0, 0, 0, 1, 1, 1]]))


def _two_movers(n=30, drop=()):
    frames = []
    for f in range(n):
        boxes = [[10 + 2 * f, 10, 30 + 2 * f, 30, .9]]
        if f not in drop:
            boxes.append([300 - 3 * f, 200, 320 - 3 * f, 220, .9])
        frames.append(np.array(boxes, dtype=np.float32))
    return frames


def test_v1_tracker_runs_and_keeps_ids():
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from src.TALOS.track.v1 import V1Tracker
    res = run_sequence(V1Tracker(), _two_movers())
    ids = np.unique(res.rows[:, 1])
    assert len(ids) == 2 and ids.min() >= 1
    assert res.rows.shape[1] == 9


def test_v1_reproduces_stale_box_bug():
    pytest.importorskip("ultralytics")
    from src.TALOS.track.v1 import V1Tracker
    frames = _two_movers(10) + [np.zeros((0, 5), np.float32)] * 3
    res = run_sequence(V1Tracker(), frames)
    last = res.rows[res.rows[:, 0] == 13]
    assert len(last) == 2  # v1 keeps reporting both tracks with no detections


def test_degrade_is_seeded_and_does_what_it_says():
    rng = np.random.default_rng(1)
    rows = []
    for f in range(1, 201):
        for k in range(5):
            x = 100 * k + f
            rows.append([f, x, 50, x + 20, 70, 1.0])
    gt = np.array(rows, dtype=np.float32)
    a = cache.degrade(gt, (1920, 1080), miss=0.2, burst=5, jitter=2, fp_per_frame=0.5, seed=0)
    b = cache.degrade(gt, (1920, 1080), miss=0.2, burst=5, jitter=2, fp_per_frame=0.5, seed=0)
    assert np.array_equal(a, b)
    clean = cache.degrade(gt, (1920, 1080), miss=0.2, burst=5, seed=0)
    kept = len(clean) / len(gt)
    assert 0.7 < kept < 0.9                     # about 20% missed
    n_fp = len(a) - len(clean)
    assert 60 < n_fp < 140                      # ~0.5 per frame over 200 frames
    assert (a[:, 5] >= 0.1).all() and (a[:, 5] <= 1).all()
