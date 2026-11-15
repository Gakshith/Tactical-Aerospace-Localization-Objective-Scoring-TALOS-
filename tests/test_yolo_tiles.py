import json
from pathlib import Path

import numpy as np
import pytest

from src.TALOS.data import make_yolo_tiles as T
from src.TALOS.data.mot import SeqInfo, write_mot, write_seqinfo


def test_tile_starts_cover_and_flush():
    assert T.tile_starts(1920, 640, 0.2) == [0, 512, 1024, 1280]
    assert T.tile_starts(1080, 640, 0.2) == [0, 440]
    assert T.tile_starts(1920, 1024, 0.2) == [0, 819, 896]
    assert T.tile_starts(500, 640, 0.2) == [0]
    for length, tile in [(1920, 640), (1080, 640), (1080, 1024), (1000, 300)]:
        s = T.tile_starts(length, tile, 0.2)
        assert s[0] == 0 and s[-1] + tile == length  # flush with both edges
        assert all(b - a <= tile for a, b in zip(s, s[1:]))  # no gaps


def test_tile_grid_full_frame():
    grid = T.tile_grid(1920, 1080, 640, 0.2)
    assert len(grid) == 8
    cover = np.zeros((1080, 1920), bool)
    for x0, y0, x1, y1 in grid:
        assert (x1 - x0, y1 - y0) == (640, 640)
        cover[y0:y1, x0:x1] = True
    assert cover.all()


def test_boxes_in_tile_threshold_and_clip():
    tile = (100, 100, 200, 200)
    boxes = np.array([
        [120, 120, 140, 140],  # fully inside
        [90, 150, 110, 170],   # 50% inside -> kept, clipped
        [85, 150, 105, 170],   # 25% inside -> dropped
        [190, 190, 230, 230],  # 1/16 inside -> dropped
        [300, 300, 310, 310],  # outside
    ], float)
    cls = np.array([0, 1, 1, 0, 0])
    b, c = T.boxes_in_tile(boxes, cls, tile, 0.5)
    assert c.tolist() == [0, 1]
    assert np.allclose(b, [[20, 20, 40, 40], [0, 50, 10, 70]])
    # tile coords never leave [0, tile size]
    assert b.min() >= 0 and b.max() <= 100


def test_yolo_lines_normalised():
    lines = T.yolo_lines(np.array([[0, 50, 10, 70]]), np.array([1]), 100, 100)
    assert lines == ["1 0.050000 0.600000 0.100000 0.200000"]


def test_black_fraction():
    img = np.full((100, 100, 3), 120, np.uint8)
    img[60:] = 3  # bottom band, JPEG-ish noise level
    assert T.black_fraction(img) == pytest.approx(0.4)
    assert T.black_fraction(np.zeros((10, 10, 3), np.uint8)) == 1.0


def test_sampled_frames():
    assert T.sampled_frames(12, 5) == [1, 6, 11]


def _make_bench(root: Path, bench: str, seq: str, rows: list, frames: int, img: np.ndarray):
    import cv2

    d = root / bench / seq
    (d / "img1").mkdir(parents=True)
    for f in range(1, frames + 1):
        cv2.imwrite(str(d / "img1" / f"{f:06d}.jpg"), img)
    write_seqinfo(d, SeqInfo(seq, frames, img.shape[1], img.shape[0], 10.0))
    write_mot(d / "gt" / "gt.txt", np.array(rows, float))


def test_build_end_to_end(tmp_path):
    pytest.importorskip("cv2")
    h, w = 400, 600
    img = np.full((h, w, 3), 128, np.uint8)
    img[300:] = 0  # black band: bottom tiles of a 200-px grid are fully black
    mot = tmp_path / "mot"
    # plane straddling the x=200 tile seam in every frame; ship only in train frames
    planes = [[f, 1, 190, 50, 20, 20, 1, 1, 1] for f in range(1, 7)]
    ships = [[f, 1, 450, 120, 6, 6, 1, 1, 1] for f in range(1, 7)]
    _make_bench(mot, "AIRMOT_plane-all", "airmot-001", planes, 6, img)
    _make_bench(mot, "AIRMOT_ship-all", "airmot-001", ships, 6, img)  # same video in both
    _make_bench(mot, "AIRMOT_plane-all", "airmot-002", planes[:3], 3, img)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"split": {"train": ["airmot-001"], "val": ["airmot-002"], "test": ["nope"]}}))

    a = T.parse_args(["--name", "t", "--mot-root", str(mot), "--out-root", str(tmp_path / "yolo"),
                      "--split-file", str(split), "--tile", "200", "--overlap", "0",
                      "--stride-train", "5", "--stride-val", "10", "--neg-ratio", "1", "--workers", "1"])
    st = T.build(a)
    out = tmp_path / "yolo" / "t"
    tr = st["splits"]["train"]
    # train frames 1, 6; 3x2 grid; bottom row (y 200-400) is 50% black so kept
    assert tr["frames_sampled"] == 2 and tr["tiles_scanned"] == 12
    # plane split 50/50 across the seam -> labelled in both tiles; ship in one
    assert tr["boxes"] == {"plane": 4, "ship": 2}
    assert tr["tiles_positive"] == 6 and tr["tiles_negative"] == 6
    assert st["splits"]["val"]["frames_sampled"] == 1
    labels = sorted((out / "labels" / "train").glob("*.txt"))
    assert len(labels) == len(list((out / "images" / "train").glob("*.jpg"))) == 12
    lab = (out / "labels" / "train" / "airmot-001_f000001_x0000_y0000.txt").read_text().split()
    assert lab[0] == "0" and float(lab[1]) == pytest.approx(195 / 200)  # clipped to x 190..200
    assert "names:\n  0: plane\n  1: ship" in (out / "data.yaml").read_text()
    with pytest.raises(SystemExit):
        T.build(a)  # refuses to overwrite without --overwrite


def test_black_tiles_dropped(tmp_path):
    pytest.importorskip("cv2")
    img = np.full((400, 400, 3), 128, np.uint8)
    img[200:] = 0  # bottom tile row fully black
    mot = tmp_path / "mot"
    _make_bench(mot, "AIRMOT_ship-all", "airmot-003", [[1, 1, 20, 20, 10, 10, 1, 1, 1]], 1, img)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"split": {"train": ["airmot-003"], "val": ["airmot-003"], "test": []}}))
    st = T.build(T.parse_args(["--name", "b", "--mot-root", str(mot), "--out-root", str(tmp_path / "y"),
                               "--split-file", str(split), "--tile", "200", "--overlap", "0",
                               "--neg-ratio", "5", "--workers", "1"]))
    tr = st["splits"]["train"]
    assert tr["tiles_dropped_black"] == {"positive": 0, "negative": 2}
    assert tr["tiles_positive"] == 1 and tr["tiles_negative"] == 1  # only the one usable negative
