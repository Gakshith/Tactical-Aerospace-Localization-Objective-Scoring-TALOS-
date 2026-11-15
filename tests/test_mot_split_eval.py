import numpy as np
import pytest

from src.TALOS.data import mot
from src.TALOS.data.split import make_split, save_split, load_split


def _gt():
    # two objects moving right for 10 frames
    rows = []
    for f in range(1, 11):
        rows.append([f, 1, 10 + 3 * f, 20, 12, 10, 1, 1, 1])
        rows.append([f, 2, 200 - 2 * f, 80, 14, 12, 1, 1, 1])
    return np.array(rows, dtype=float)


def test_mot_roundtrip(tmp_path):
    p = tmp_path / "gt.txt"
    mot.write_mot(p, _gt())
    back = mot.read_mot(p)
    assert back.shape == (20, 9)
    assert np.allclose(back[:, :6], _gt()[:, :6])


def test_box_conversions():
    b = np.array([[1.0, 2.0, 5.0, 10.0]])
    assert np.allclose(mot.xywh_to_xyxy(mot.xyxy_to_xywh(b)), b)


def test_seqinfo_roundtrip(tmp_path):
    info = mot.SeqInfo("s1", 10, 1024, 1024, 10.0)
    mot.write_seqinfo(tmp_path, info)
    assert mot.read_seqinfo(tmp_path) == info


def test_split_disjoint_and_deterministic():
    vids = [f"v{i:02d}" for i in range(20)]
    a, b = make_split(vids, seed=3), make_split(vids, seed=3)
    assert a == b
    allv = a["train"] + a["val"] + a["test"]
    assert sorted(allv) == vids and len(set(allv)) == 20
    assert len(a["test"]) == 3 and len(a["val"]) == 3


def test_split_respects_pinned_test():
    vids = [f"v{i}" for i in range(10)]
    s = make_split(vids, fixed={"test": ["v1", "v2"]})
    assert s["test"] == ["v1", "v2"]
    assert not set(s["train"] + s["val"]) & {"v1", "v2"}


def test_split_is_frozen(tmp_path):
    s = make_split(["a", "b", "c", "d"])
    save_split("x", s, {"seed": 0}, root=tmp_path)
    assert load_split("x", root=tmp_path) == s
    with pytest.raises(FileExistsError):
        save_split("x", s, {"seed": 0}, root=tmp_path)


def _layout(root, rows_by_tracker, gt):
    bench, split, seq = "T", "test", "seq1"
    sd = root / "gt" / f"{bench}-{split}" / seq
    mot.write_mot(sd / "gt" / "gt.txt", gt)
    mot.write_seqinfo(sd, mot.SeqInfo(seq, 10, 256, 128, 10.0))
    for name, rows in rows_by_tracker.items():
        mot.write_mot(root / "trk" / f"{bench}-{split}" / name / "data" / f"{seq}.txt", rows)
    return root / "gt", root / "trk", bench, split, [seq]


def test_trackeval_perfect_and_idswitch(tmp_path):
    pytest.importorskip("trackeval")
    from src.TALOS.eval.trackeval_runner import run_trackeval

    gt = _gt()
    swapped = gt.copy()
    late = swapped[:, 0] > 5
    swapped[late, 1] = 3 - swapped[late, 1]  # ids 1<->2 swap halfway
    args = _layout(tmp_path, {"perfect": gt, "swap": swapped}, gt)
    res = run_trackeval(*args[:4], ["perfect", "swap"], args[4], num_cores=1)
    assert res["perfect"]["HOTA"] == pytest.approx(100.0)
    assert res["perfect"]["IDF1"] == pytest.approx(100.0)
    assert res["swap"]["IDSW"] == 2
    assert res["swap"]["IDF1"] < 60
