#!/usr/bin/env bash
# Fit the threat model's one free parameter (white-acceleration std, px/s^2) on
# train trajectories by Brier score, then report the chosen value on val.
#   DATA=<data root> scripts/fit_threat_noise.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p results/threat/fit
for a in 5 10 20 40 80; do
  python -m src.TALOS.threat.evaluate --splits train --accel-std "$a" \
    --out "results/threat/fit/train_accel${a}.json" > /dev/null
  python - "$a" <<'PY'
import json, sys
a = sys.argv[1]
d = json.load(open(f"results/threat/fit/train_accel{a}.json"))
c, v = d["v2_calibration"], d["v2_entry_prob"]
print(f"accel_std={a:>3}  brier={c['brier']:.5f}  ece={c['ece']:.4f}  auroc={v['auroc']:.4f}")
PY
done
