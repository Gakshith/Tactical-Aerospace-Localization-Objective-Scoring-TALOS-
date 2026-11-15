---
title: TALOS Satellite Tracking
emoji: 🛰️
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: Aircraft tracking and zone-entry warnings in satellite video
---

# TALOS

Detects, tracks and threat-ranks aircraft and ships in satellite video. Runs on CPU with ONNX Runtime.
Code and full results: https://github.com/Gakshith/Tactical-Aerospace-Localization-Objective-Scoring-TALOS-

This Space is licensed CC BY-NC-SA 4.0 because it ships annotated clips from VISO, which carry that
license (Yin et al., IEEE TGRS 2021, https://github.com/QingyongHu/VISO). The code is MIT. The detector was
trained on AIR-MOT (He et al., IEEE TGRS 2022, https://github.com/HeQibin/TGraM); no AIR-MOT frames are
included.
