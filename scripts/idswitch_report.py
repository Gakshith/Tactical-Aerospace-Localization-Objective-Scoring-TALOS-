"""Where do a tracker's ID switches happen? Per sequence and per GT track,
match tracker boxes to GT (IoU >= 0.5, Hungarian per frame) and list every
frame where a GT object's matched tracker ID changes.

    python scripts/idswitch_report.py <gt.txt> <tracker.txt>
"""
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, ".")
from src.TALOS.data.mot import read_mot  # noqa: E402
from src.TALOS.track.calibrate import iou_matrix  # noqa: E402


def xyxy(r):
    return np.column_stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]])


gt, tr = read_mot(sys.argv[1]), read_mot(sys.argv[2])
last, switches, gaps = {}, [], defaultdict(int)
for f in np.unique(gt[:, 0]):
    g, t = gt[gt[:, 0] == f], tr[tr[:, 0] == f]
    if not len(t):
        continue
    iou = iou_matrix(xyxy(g), xyxy(t))
    r, c = linear_sum_assignment(-iou)
    for i, j in zip(r, c):
        if iou[i, j] < 0.5:
            continue
        gid, tid = int(g[i, 1]), int(t[j, 1])
        if gid in last and last[gid][0] != tid:
            switches.append((int(f), gid, last[gid][0], tid, int(f - last[gid][1])))
        last[gid] = (tid, f)
print(f"{len(switches)} switches")
for s in switches[:25]:
    print(f"  frame {s[0]:4d}  gt {s[1]:3d}: id {s[2]} -> {s[3]}  (gap since last match: {s[4]} frames)")
