"""Freeze the AIR-MOT train/val/test split (by video, stratified).

No official split exists for AIR-MOT-100. Videos are split 70/15/15 inside
three strata so every split gets its share of the rare cases:
  moving-plane  videos with an aircraft track that moves > 100 px
  plane         other videos with aircraft
  ship-only     videos with ships and no aircraft

    python -m src.TALOS.data.make_airmot_split data_meta/airmot_tracks.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.TALOS.data.split import make_split, save_split

SEED = 0
MOVING_PX = 100.0


def strata(tracks: pd.DataFrame) -> dict[str, list[str]]:
    seq = lambda v: f"airmot-{int(v):03d}"
    planes = tracks[tracks.cls == 1]
    moving = set(planes[planes.net_disp_px > MOVING_PX].video)
    plane_v = set(planes.video) - moving
    ship_only = set(tracks.video) - moving - plane_v
    return {"moving-plane": sorted(map(seq, moving)), "plane": sorted(map(seq, plane_v)),
            "ship-only": sorted(map(seq, ship_only))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks_csv")
    a = ap.parse_args()
    groups = strata(pd.read_csv(a.tracks_csv))
    split = {"train": [], "val": [], "test": []}
    for vids in groups.values():
        part = make_split(vids, seed=SEED)
        for k in split:
            split[k] += part[k]
    split = {k: sorted(v) for k, v in split.items()}
    meta = {"seed": SEED, "fractions": [0.7, 0.15, 0.15], "unit": "video",
            "strata": {k: len(v) for k, v in groups.items()},
            "strata_members": groups,
            "note": "AIR-MOT-100 (TGraM issue #9 release); 31 unannotated videos excluded"}
    path = save_split("airmot", split, meta)
    print(path, {k: len(v) for k, v in split.items()})
    for name, vids in groups.items():
        print(name, {k: len(set(vids) & set(split[k])) for k in split})


if __name__ == "__main__":
    main()
