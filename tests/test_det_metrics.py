import numpy as np
import pytest

from src.TALOS.eval import det_metrics as dm


def test_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert np.allclose(dm.iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


def test_match_frame_greedy_by_conf():
    gts = np.array([[0, 0, 10, 10], [10, 0, 20, 10]], float)
    # the lower-conf det overlaps GT0 better, but the higher-conf one goes first
    dets = np.array([[1, 0, 11, 10, 0.4], [2, 0, 12, 10, 0.9]], float)
    # 0.9 det: IoU .667 with GT0 -> takes it; 0.4 det: GT0 taken, IoU .053 with GT1 -> FP
    tp, gc = dm.match_frame(dets, gts)
    assert tp.tolist() == [False, True]
    assert gc.tolist() == [0.9, -1.0]


def test_hand_checked_sequence():
    # frame 1: GT A 10px (lt16), GT B 40px (gt32); frame 2: GT C 20px (16to32)
    gt = np.array([[1, 1, 0, 0, 10, 10, 1, 1, 1],
                   [1, 2, 100, 100, 40, 40, 1, 1, 1],
                   [2, 3, 0, 0, 20, 20, 1, 1, 1]], float)
    dets = np.array([[1, 0, 0, 10, 10, 0.9],    # TP on A
                     [1, 1, 0, 11, 10, 0.8],    # duplicate of A -> FP
                     [1, 200, 200, 210, 210, 0.3],  # FP
                     [2, 0, 0, 20, 20, 0.4]], float)  # TP on C; B never found
    r = dm.summarize(*dm.match_sequence(dets, gt, n_frames=2, min_conf=0.1), at_conf=0.5)
    assert r["gt_boxes"] == 3 and r["dets_all"] == 4
    at = r["at_conf_0.5"]
    assert (at["tp"], at["fp"], at["fn"]) == (1, 1, 2)
    assert at["precision"] == pytest.approx(0.5) and at["recall"] == pytest.approx(1 / 3)
    # ranked: TP .9, FP .8, TP .4, FP .3 -> AP = 1/3 * 1 + 1/3 * 2/3
    assert r["ap50"] == pytest.approx(5 / 9)
    s = r["recall_by_size"]
    assert s["lt16"]["recall_at_0.5"] == 1.0
    assert s["16to32"]["recall_at_0.5"] == 0.0 and s["16to32"]["recall_at_min_conf"] == 1.0
    assert s["gt32"]["recall_at_0.5"] == 0.0


def test_ap_edge_cases():
    assert dm.ap_all_point(np.array([True]), np.array([0.9]), 1) == 1.0
    assert dm.ap_all_point(np.zeros(0, bool), np.zeros(0), 5) == 0.0
    assert np.isnan(dm.ap_all_point(np.zeros(0, bool), np.zeros(0), 0))
