"""TiledDetector with a fake model (no GPU, no ultralytics).

The fake "sees" only pixels: a frame is an (H, W, n_obj) int array, layer k
holds object k's id inside its box and 0 elsewhere, so objects may overlap.
For any crop the fake reports, per visible object, the bounding box of its
visible pixels in crop coordinates, with that object's conf and class, which
is what a perfect detector does on a cut object.
"""
import numpy as np
import pytest

from src.TALOS.detect.tiled import TiledDetector, axis_starts, cut_flags, merge_tiles, tile_grid


class FakeModel:
    def __init__(self, objs, min_side=2):
        self.objs = objs          # list of (x1, y1, x2, y2, conf, cls), frame px, x2/y2 exclusive
        self.min_side = min_side  # smaller visible fragments are not detected
        self.calls = []

    def __call__(self, images):
        self.calls.append(len(images))
        out = []
        for im in images:
            rows = []
            for k in range(im.shape[2]):
                ys, xs = np.nonzero(im[:, :, k])
                if len(xs) == 0:
                    continue
                x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                if min(x2 - x1, y2 - y1) < self.min_side:
                    continue
                _, _, _, _, c, cl = self.objs[k]
                rows.append([x1, y1, x2, y2, c, cl])
            out.append(np.array(rows, np.float32).reshape(-1, 6))
        return out


def render(w, h, objs):
    fr = np.zeros((h, w, max(len(objs), 1)), np.int32)
    for k, (x1, y1, x2, y2, *_ ) in enumerate(objs):
        fr[y1:y2, x1:x2, k] = k + 1
    return fr


def det(objs, w=1920, h=1080, full=False, full_min_side=40, **kw):
    fake = FakeModel(objs)
    full_fake = FakeModel(objs, min_side=full_min_side) if full else None
    td = TiledDetector(tile=kw.pop("tile", 640), overlap=kw.pop("overlap", 0.2), batch=kw.pop("batch", 64),
                       with_full_frame=full, backend=fake, full_backend=full_fake, **kw)
    return td, fake, render(w, h, objs)


def boxes_of(res):
    return sorted(tuple(np.round(r[:4]).astype(int).tolist()) + (round(float(r[4]), 3), int(r[5])) for r in res)


# ---------- grid ----------

def test_grid_1080p_layout():
    assert axis_starts(1920, 640, 0.2) == ([0, 512, 1024, 1280], 640)
    assert axis_starts(1080, 640, 0.2) == ([0, 440], 640)
    g = tile_grid(1920, 1080, 640, 0.2)
    assert len(g) == 8 and (g[:, 2] - g[:, 0] == 640).all() and (g[:, 3] - g[:, 1] == 640).all()


@pytest.mark.parametrize("w,h,tile,ov", [(1920, 1080, 640, 0.2), (1920, 1080, 512, 0.25), (1000, 300, 640, 0.2),
                                         (1280, 640, 640, 0.0), (777, 555, 100, 0.5), (50, 40, 640, 0.2)])
def test_grid_covers_frame_last_tile_flush(w, h, tile, ov):
    g = tile_grid(w, h, tile, ov)
    cover = np.zeros((h, w), int)
    for x0, y0, x1, y1 in g:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        assert x1 - x0 == min(tile, w) and y1 - y0 == min(tile, h)
        cover[y0:y1, x0:x1] += 1
    assert cover.min() >= 1
    assert g[:, 2].max() == w and g[:, 3].max() == h  # flush with the far edges
    if w > tile:  # neighbours overlap by at least the requested fraction
        xs = sorted(set(g[:, 0].tolist()))
        assert all(xs[i] + tile - xs[i + 1] >= int(round(tile * ov)) for i in range(len(xs) - 1))


def test_no_duplicate_tile_when_exact_fit():
    assert axis_starts(1280, 640, 0.0) == ([0, 640], 640)


# ---------- coordinates ----------

def test_object_inside_interior_tile_maps_back_exactly():
    # x 1700..1712 lies only in the last (flush, x0=1280) column; y 900..915 only in the bottom row (y0=440)
    objs = [(1700, 900, 1712, 915, 0.8, 0)]
    td, _, fr = det(objs)
    assert boxes_of(td.predict([fr])[0]) == [(1700, 900, 1712, 915, 0.8, 0)]


def test_every_location_maps_back(rng=np.random.default_rng(0)):
    objs = []
    for _ in range(40):
        x, y = int(rng.integers(0, 1900)), int(rng.integers(0, 1060))
        s = int(rng.integers(4, 20))
        objs.append((x, y, min(x + s, 1920), min(y + s, 1080), 0.5, 0))
    # keep them apart so overlapping-object effects do not enter this test
    keep = []
    for o in objs:
        if all(o[2] + 3 < p[0] or p[2] + 3 < o[0] or o[3] + 3 < p[1] or p[3] + 3 < o[1] for p in keep):
            keep.append(o)
    td, _, fr = det(keep)
    assert boxes_of(td.predict([fr])[0]) == sorted(o[:4] + (0.5, 0) for o in keep)


# ---------- straddling objects ----------

def test_object_across_tile_edge_gives_one_full_box():
    # tile 0 ends at x=640; tile 1 (512..1152) sees the whole plane. Tile 0 sees a 10-px fragment.
    objs = [(630, 100, 660, 130, 0.7, 0)]
    td, _, fr = det(objs)
    assert boxes_of(td.predict([fr])[0]) == [(630, 100, 660, 130, 0.7, 0)]


def test_fragment_with_higher_conf_still_loses_geometry_keeps_conf():
    objs = [(630, 100, 660, 130, 0.7, 0)]
    fake = FakeModel(objs)

    def biased(images):  # the fragment (touching the crop's right edge) scores higher
        out = fake(images)
        for im, d in zip(images, out):
            for r in d:
                if r[2] >= im.shape[1] - 1:
                    r[4] = 0.95
        return out
    td = TiledDetector(tile=640, overlap=0.2, backend=biased)
    assert boxes_of(td.predict([render(1920, 1080, objs)])[0]) == [(630, 100, 660, 130, 0.95, 0)]


def test_object_in_overlap_seen_whole_twice_gives_one_box():
    objs = [(560, 300, 590, 330, 0.6, 1)]  # inside both tile 0 (0..640) and tile 1 (512..1152)
    td, _, fr = det(objs)
    assert boxes_of(td.predict([fr])[0]) == [(560, 300, 590, 330, 0.6, 1)]


def test_object_wider_than_overlap_is_unioned():
    # 200-px ship over x 500..700: cut by tile 0's right edge (640) and tile 1's left edge (512)
    objs = [(500, 50, 700, 90, 0.9, 1)]
    td, _, fr = det(objs)
    assert boxes_of(td.predict([fr])[0]) == [(500, 50, 700, 90, 0.9, 1)]


def test_long_object_fragments_with_small_overlap_are_unioned():
    # 600-px object over x 300..900: fragments [300,640] (cut R) and [512,900] (cut L) share only
    # the 128-px overlap, so neither IoU (0.21) nor IoS (0.33) joins them; only the continuation rule does
    objs = [(300, 50, 900, 90, 0.9, 1)]
    td, _, fr = det(objs)
    assert boxes_of(td.predict([fr])[0]) == [(300, 50, 900, 90, 0.9, 1)]


def test_cut_flags_ignore_frame_borders():
    w, h = 1920, 1080
    first, last = np.array([0, 0, 640, 640]), np.array([1280, 440, 1920, 1080])
    b = np.array([[0, 0, 640, 640]], float)  # touches every side of the first tile
    assert cut_flags(b, first, w, h, 3).tolist() == [[False, False, True, True]]
    b = np.array([[1280, 440, 1920, 1080]], float)
    assert cut_flags(b, last, w, h, 3).tolist() == [[True, True, False, False]]
    b = np.array([[1284, 444, 1915, 1075]], float)  # 4 px from the interior edges: margin 3 -> not cut
    assert cut_flags(b, last, w, h, 3).tolist() == [[False, False, False, False]]


def test_object_over_2x2_tile_corner_is_one_box():
    # 160x160 frame, tile 100, overlap 0.2 -> starts 0, 60 on both axes (overlap 60..100)
    objs = [(40, 40, 120, 120, 0.8, 0)]
    td, _, fr = det(objs, w=160, h=160, tile=100, overlap=0.2)
    assert len(td.grid(160, 160)) == 4
    assert boxes_of(td.predict([fr])[0]) == [(40, 40, 120, 120, 0.8, 0)]


def test_full_frame_pass_catches_large_object():
    objs = [(400, 300, 800, 700, 0.5, 1), (100, 100, 110, 110, 0.9, 0)]
    td, _, fr = det(objs, full=True)
    assert boxes_of(td.predict([fr])[0]) == sorted([(400, 300, 800, 700, 0.5, 1), (100, 100, 110, 110, 0.9, 0)])


def test_object_at_frame_border_is_not_treated_as_cut():
    objs = [(0, 0, 12, 9, 0.4, 0), (1910, 1070, 1920, 1080, 0.4, 0)]
    td, _, fr = det(objs)
    assert boxes_of(td.predict([fr])[0]) == sorted([(0, 0, 12, 9, 0.4, 0), (1910, 1070, 1920, 1080, 0.4, 0)])


# ---------- class awareness and dense scenes ----------

def _merge(rows, cls, src, cut):
    r = np.array(rows, float)
    return merge_tiles(r[:, :4], r[:, 4], np.array(cls), np.array(src), np.array(cut, bool))


@pytest.mark.parametrize("same", [True, False])
def test_class_aware_containment(same):
    # a fragment cut at x=640 (tile 0) lies inside a whole box from tile 1
    m = _merge([[600, 100, 700, 200, .9], [630, 120, 640, 150, .7]], [1, 1 if same else 0], [1, 0],
               [[0, 0, 0, 0], [0, 0, 1, 0]])
    assert len(m) == (1 if same else 2)


@pytest.mark.parametrize("same", [True, False])
def test_class_aware_duplicates(same):
    m = _merge([[100, 100, 130, 130, .9], [101, 100, 131, 130, .8]], [0, 0 if same else 1], [0, 1],
               [[0, 0, 0, 0], [0, 0, 0, 0]])
    assert len(m) == (1 if same else 2)


@pytest.mark.parametrize("same", [True, False])
def test_class_aware_union(same):
    m = _merge([[500, 50, 640, 90, .9], [512, 50, 700, 90, .8]], [1, 1 if same else 0], [0, 1],
               [[0, 0, 1, 0], [1, 0, 0, 0]])
    assert len(m) == (1 if same else 2)
    if same:
        assert m[0, :4].tolist() == [500, 50, 700, 90]


def test_two_objects_detector_level_keep_classes():
    objs = [(600, 100, 700, 200, 0.9, 1), (630, 120, 660, 150, 0.7, 0)]
    td, _, fr = det(objs)
    assert boxes_of(td.predict([fr])[0]) == sorted([(600, 100, 700, 200, 0.9, 1), (630, 120, 660, 150, 0.7, 0)])


def test_same_class_fragment_inside_other_object_is_absorbed():
    # control for the test above: same layout, same class -> the fragment vanishes into the bigger box
    objs = [(600, 100, 700, 200, 0.9, 1), (630, 120, 660, 150, 0.7, 1)]
    td, _, fr = det(objs)
    assert (630, 120, 660, 150, 0.7, 1) in boxes_of(td.predict([fr])[0])  # seen whole by tile 1: kept
    # ...but a box that is only ever seen cut and lies inside another is absorbed
    merged = merge_tiles(np.array([[600, 100, 700, 200], [630, 120, 640, 150]], float), np.array([.9, .7]),
                         np.array([1, 1]), np.array([1, 0]), np.array([[0, 0, 0, 0], [0, 0, 1, 0]], bool))
    assert len(merged) == 1 and merged[0, 4] == pytest.approx(0.9)


def test_overlapping_same_class_neighbours_in_one_tile_survive():
    # IoU 0.6 between two moored ships, both well inside tile 0: same source -> never merged
    objs = [(100, 100, 140, 120, 0.8, 1), (110, 100, 150, 120, 0.7, 1)]
    td, _, fr = det(objs)
    assert len(td.predict([fr])[0]) == 2


def test_overlapping_neighbours_in_overlap_zone_survive():
    # same pair inside the 512..640 overlap: each tile reports both; the result is still two boxes
    objs = [(540, 100, 580, 120, 0.8, 1), (550, 100, 590, 120, 0.7, 1)]
    td, _, fr = det(objs)
    assert boxes_of(td.predict([fr])[0]) == sorted([(540, 100, 580, 120, 0.8, 1), (550, 100, 590, 120, 0.7, 1)])


# ---------- batching ----------

def test_batches_tiles_across_frames_and_keeps_order():
    a = [(630, 100, 660, 130, 0.7, 0)]
    b = [(1700, 900, 1712, 915, 0.8, 1)]
    fa, fb = render(1920, 1080, a), render(1920, 1080, b)
    # separate models per frame content are not possible with one fake, so give both objects
    # distinct layers and blank one in each frame
    objs = a + b
    frA = np.concatenate([fa, np.zeros_like(fa)], axis=2)
    frB = np.concatenate([np.zeros_like(fb), fb], axis=2)
    fake = FakeModel(objs)
    td = TiledDetector(tile=640, overlap=0.2, batch=5, backend=fake)
    res = td.predict([frA, frB, frA])
    assert fake.calls == [5, 5, 5, 5, 4]  # 24 tiles in calls of <= 5, frames mixed in a call
    assert boxes_of(res[0]) == boxes_of(res[2]) == [(630, 100, 660, 130, 0.7, 0)]
    assert boxes_of(res[1]) == [(1700, 900, 1712, 915, 0.8, 1)]
    assert td.timing["frames"] == 3 and td.timing["tiles"] == 24


def test_empty_frame():
    td, _, _ = det([])
    assert td.predict([np.zeros((1080, 1920, 1), np.int32)])[0].shape == (0, 6)
