"""Print the cumulative TALOS-Track ablation params with a calibration applied.

    python scripts/ablation_params.py configs/ablation/talos_cumulative.json results/calib/<file>.json
"""
import json
import sys

params = json.load(open(sys.argv[1]))
if len(sys.argv) > 2:
    cal = json.load(open(sys.argv[2]))
    for p in params.values():
        p.update(meas_pos_min=cal["meas_pos_min"], meas_pos_frac=cal["meas_pos_frac"])
print(json.dumps(params))
