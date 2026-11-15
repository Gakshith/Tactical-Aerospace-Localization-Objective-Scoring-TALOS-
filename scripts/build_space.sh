#!/usr/bin/env bash
# Assemble the Hugging Face Space in build/space from the repo. Run on Polaris, where the
# model and the pre-rendered examples live:
#   cd ~/talos/repo && bash scripts/build_space.sh [OUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-$HOME/talos/build/space}"
TALOS_DIR="${TALOS_DIR:-$HOME/talos}"
rm -rf "$OUT"; mkdir -p "$OUT"/{models,results,configs}
cp space/app.py space/requirements.txt space/packages.txt space/README.md "$OUT/"
# the pipeline code only: detection (no torch at run time), tracking, threat, pipeline
for f in src/__init__.py src/TALOS/__init__.py \
         src/TALOS/data/__init__.py src/TALOS/data/mot.py \
         src/TALOS/detect/__init__.py src/TALOS/detect/tiled.py src/TALOS/detect/onnx_backend.py src/TALOS/detect/cache.py \
         src/TALOS/track/__init__.py src/TALOS/track/base.py src/TALOS/track/talos_track.py \
         src/TALOS/threat/__init__.py src/TALOS/threat/model.py \
         src/TALOS/pipeline/__init__.py src/TALOS/pipeline/run.py; do
  mkdir -p "$OUT/$(dirname "$f")"; cp "$f" "$OUT/$f"
done
cp LICENSE "$OUT/LICENSE-code-MIT"
cp configs/talos_v2.json "$OUT/configs/"
cp "$TALOS_DIR/weights/export/y26s_t1024_u8.onnx" "$OUT/models/talos_y26s_t1024.onnx"
cp results/final-test-y26s1024-plane/metrics.json "$OUT/results/tracking-test-plane.json"
cp results/final-test-y26s1024-ship/metrics.json "$OUT/results/tracking-test-ship.json"
cp results/threat/d4-threat-val-accel1.json "$OUT/results/threat-airmot-val.json"
cp results/threat/d4-threat-viso-cross.json "$OUT/results/threat-viso.json"
cp results/speed/detector_speed.json "$OUT/results/"
cp -r "$TALOS_DIR/space_examples" "$OUT/examples"
cat > "$OUT/.gitattributes" <<'GA'
*.onnx filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text
GA
# no private paths in what gets published
if grep -rIl "/Users/[0-9]" "$OUT" --include=*.json --include=*.py --include=*.csv --include=*.jsonl; then
  echo "private paths found above; strip them before publishing" >&2; exit 1
fi
du -sh "$OUT"; find "$OUT" -type f | wc -l
