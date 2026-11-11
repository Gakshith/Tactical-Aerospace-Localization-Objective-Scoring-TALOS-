import numpy as np
import pytest

pytest.importorskip("trackers")
pytest.importorskip("supervision")

from src.TALOS.track import external  # noqa: E402,F401  (fills REGISTRY)
from src.TALOS.track.base import get, run_sequence  # noqa: E402

NAMES = ["sort", "bytetrack", "ocsort", "botsort"]
FPS = 10.0
N = 40
DROP = (20, 21, 22)          # object B missing from these frames (0-based)
SPURIOUS_FRAME = 15          # a one-frame false positive far from both objects
SPURIOUS = [1500, 900, 1520, 920, 0.95]


def _box(x, y, s=20):
    return np.array([x, y, x + s, y + s], dtype=float)


def _gt(f):
    a = _box(100 + 3 * f, 100 + 1 * f)      # moves right/down 3,1 px per frame
    b = _box(600 - 2 * f, 400 + 2 * f)      # moves left/down 2,2 px per frame
    return a, b


def _frames():
    frames = []
    for f in range(N):
        a, b = _gt(f)
        rows = [[*a, 0.9]]
        if f not in DROP:
            rows.append([*b, 0.85])
        if f == SPURIOUS_FRAME:
            rows.append(SPURIOUS)
        frames.append(np.array(rows, dtype=np.float32))
    return frames


def _iou(p, q):
    w = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
    h = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = w * h
    return inter / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter)


def _run(name):
    trk = get(name)(fps=FPS)
    outs = [trk.update(d) for d in _frames()]
    return trk, outs


@pytest.mark.parametrize("name", NAMES)
def test_output_contract(name):
    _, outs = _run(name)
    for out in outs:
        assert isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] == 6
        assert np.issubdtype(out.dtype, np.floating)
        if len(out):
            ids = out[:, 4]
            assert (ids >= 1).all() and np.allclose(ids, np.round(ids))
            assert len(np.unique(ids)) == len(ids)   # one row per track per frame


@pytest.mark.parametrize("name", NAMES)
def test_two_objects_keep_one_id_each_through_dropout(name):
    _, outs = _run(name)
    ids = {"a": [], "b": []}
    frames_seen = {"a": set(), "b": set()}
    for f, out in enumerate(outs):
        a, b = _gt(f)
        for row in out:
            if _iou(row[:4], a) > 0.5:
                key = "a"
            elif _iou(row[:4], b) > 0.5:
                key = "b"
            else:
                pytest.fail(f"{name}: frame {f} output {row} matches no object (spurious det tracked?)")
            ids[key].append(int(row[4]))
            frames_seen[key].add(f)
            # conf is the matched detection's own confidence
            assert row[5] == pytest.approx(0.9 if key == "a" else 0.85, abs=1e-6)
    # Library default lost_track_buffer=30 "30-fps frames" -> 10 frames at 10 fps,
    # so a 3-frame dropout must not cost an ID for any of the four.
    assert len(set(ids["a"])) == 1, f"{name}: A ids {sorted(set(ids['a']))}"
    assert len(set(ids["b"])) == 1, f"{name}: B ids {sorted(set(ids['b']))}"
    assert ids["a"][0] != ids["b"][0]
    # nothing reported for B while it is missing (we do not emit coasting boxes)
    assert not frames_seen["b"] & set(DROP)
    # both are tracked again well after the dropout
    late = set(range(30, N))
    assert late <= frames_seen["a"] and late <= frames_seen["b"]


@pytest.mark.parametrize("name", NAMES)
def test_tentative_tracks_not_reported(name):
    # A lone detection seen once never gets confirmed, so it is never output.
    trk = get(name)(fps=FPS)
    empty = np.zeros((0, 5), dtype=np.float32)
    for _ in range(5):                              # get past start-up special cases
        trk.update(empty)
    for _ in range(5):
        trk.update(np.array([[10, 10, 30, 30, 0.9]], dtype=np.float32))
    out = trk.update(np.array([SPURIOUS], dtype=np.float32))
    assert all(_iou(r[:4], SPURIOUS[:4]) < 0.5 for r in out)


@pytest.mark.parametrize("name", NAMES)
def test_empty_frames(name):
    trk = get(name)(fps=FPS)
    empty = np.zeros((0, 5), dtype=np.float32)
    assert trk.update(empty).shape == (0, 6)                # before any track exists
    frames = _frames()
    for d in frames[:10]:
        trk.update(d)
    for _ in range(3):
        assert trk.update(empty).shape == (0, 6)            # while tracks are coasting
    res = run_sequence(get(name)(fps=FPS), [empty] * 3 + frames + [empty] * 3)
    assert res.rows.shape[1] == 9 and res.n_frames == N + 6
    assert len(res.rows) and (res.rows[:, 1] >= 1).all()


def test_factories_pass_fps_and_resolve_params():
    trk = get("sort")(fps=FPS)
    assert trk.params["frame_rate"] == FPS
    assert trk.impl.maximum_frames_without_update == 10      # 30 @30fps -> 10 @10fps
    trk = get("bytetrack")(fps=FPS, minimum_iou_threshold=0.05)
    assert trk.impl.minimum_iou_threshold == 0.05
    bot = get("botsort")(fps=FPS)
    assert bot.params["enable_cmc"] is False and bot.impl.cmc is None
    assert bot.params["reid_model"] is None and bot.impl.reid_model is None
    assert get("ocsort")(fps=FPS).impl.enable_cmc is False
    with pytest.raises(TypeError):
        get("sort")(fps=FPS, not_a_param=1)
    with pytest.raises(TypeError):
        get("sort")(fps=FPS, frame_rate=30)
    with pytest.raises(ValueError):
        get("botsort")(fps=FPS, enable_cmc=True)
    with pytest.raises(ValueError):
        get("botsort")(fps=FPS, reid_model="osnet")
