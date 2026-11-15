import numpy as np

from src.TALOS.data.convert_viso import load_gt, repair_duplicates


def test_both_formats_parse_to_xywh(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("1,1,10,20,30,40,1,-1,-1,-1\n2,1,12,20,30,40,1,-1,-1,-1\n")
    b = tmp_path / "b.txt"
    b.write_text("1 1 10 20 40 60 -1 -1 -1 -1\r\n2 1 12 20 42 60 -1 -1 -1 -1\r\n")
    ra, fa = load_gt(a)
    rb, fb = load_gt(b)
    assert (fa, fb) == ("A", "B")
    assert np.allclose(ra, rb)            # same boxes expressed two ways
    assert np.allclose(rb[0, 2:], [10, 20, 30, 40])


def test_duplicate_row_far_from_track_is_dropped():
    rows = np.array([[f, 1, 100 + 2 * f, 50, 20, 20] for f in range(1, 11)], float)
    stray = np.array([[5, 1, 600, 400, 20, 20]], float)   # mislabelled id, far away
    fixed, dropped = repair_duplicates(np.vstack([rows, stray]))
    assert len(fixed) == 10 and len(dropped) == 1
    assert dropped[0]["dropped_xywh"][0] == 600
