#!/usr/bin/env bash
#SBATCH --job-name=talos-tune
#SBATCH --partition=cpucluster
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
# Equal-budget tracker tuning on val (src/TALOS/eval/tune.py). Submit from the repo root:
#   cd ~/talos/repo && BENCH=AIRMOT_plane DETECTOR=y26m_t640 RUN_ID=tune-... \
#     CALIB=results/calib/y26m_t640-plane-val.json sbatch --export=ALL jobs/tune.sbatch
# Env: BENCH, DETECTOR, RUN_ID (required); TRACKERS (sort bytetrack ocsort botsort talos),
#      BUDGET (30), SEED (0), CALIB (noise calibration JSON held fixed for talos).
source "$HOME/talos/repo/jobs/common.sh"
: "${BENCH:?}" "${DETECTOR:?}" "${RUN_ID:?}"
FIXED='{}'
if [ -n "${CALIB:-}" ]; then
  FIXED=$(python -c "import json,sys;c=json.load(open(sys.argv[1]));print(json.dumps({'talos':{'meas_pos_min':c['meas_pos_min'],'meas_pos_frac':c['meas_pos_frac']}}))" "$CALIB")
fi
python -m src.TALOS.eval.tune --bench "$BENCH" --detector "$DETECTOR" \
  --trackers ${TRACKERS:-sort bytetrack ocsort botsort talos} \
  --budget "${BUDGET:-30}" --seed "${SEED:-0}" --fixed "$FIXED" --run-id "$RUN_ID"
echo "TUNE DONE"
