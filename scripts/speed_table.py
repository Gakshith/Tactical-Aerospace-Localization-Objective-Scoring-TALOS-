"""Collect the runtime comparison written by jobs/export_parity.sbatch into one table.

    python scripts/speed_table.py --root results/speed --ref rt_torch_fp32 --out results/speed/detector_speed.json

For every ``<root>/<variant>/`` with meta.json + plane.json + ship.json: detector ms/frame
(tiles + merge, the same 1920x1080 val frames for every variant), AP50 and box counts per
class, and parity against ``--ref``: |dAP50| <= 0.005 and box count within 1% per class.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

AP_TOL, BOX_TOL = 0.005, 0.01


def load(d: Path, split: str) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    run = meta["runs"].get(f"AIRMOT_plane/{split}") or next(iter(meta["runs"].values()))
    t = run["job_timing"]
    row = {"variant": d.name, "backend": meta.get("backend", "torch"),
           "detail": meta.get("backend_detail", {}).get("provider") or ("fp16" if t.get("half") else "fp32"),
           "gpu": meta.get("gpu"), "frames": t["total_frames"],
           "detect_ms_per_frame": t["mean_ms_per_frame_predict"],
           "fps_detect": round(1000 / t["mean_ms_per_frame_predict"], 1),
           "classes": {}}
    if meta.get("backend_detail"):
        row["fp16"] = meta["backend_detail"].get("fp16")
    for b in ("plane", "ship"):
        r = json.loads((d / f"{b}.json").read_text())[f"AIRMOT_{b}"]["splits"][split]
        row["classes"][b] = {"ap50": r["ap50"], "dets_all": r["dets_all"]}
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = [load(d, a.split) for d in sorted(a.root.iterdir())
            if d.is_dir() and all((d / f).exists() for f in ("meta.json", "plane.json", "ship.json"))]
    ref = next(r for r in rows if r["variant"] == a.ref)
    for r in rows:
        par = {}
        for b, c in r["classes"].items():
            rc = ref["classes"][b]
            d_ap = round(c["ap50"] - rc["ap50"], 5)
            d_box = round((c["dets_all"] - rc["dets_all"]) / max(rc["dets_all"], 1), 5)
            par[b] = {"d_ap50": d_ap, "d_boxes_frac": d_box,
                      "ok": abs(d_ap) <= AP_TOL and abs(d_box) <= BOX_TOL}
        r["parity_vs_ref"] = par
        r["parity_ok"] = all(p["ok"] for p in par.values())
        r["speedup_vs_ref"] = round(ref["detect_ms_per_frame"] / r["detect_ms_per_frame"], 2)
    out = {"split": a.split, "ref": a.ref, "tolerance": {"ap50_abs": AP_TOL, "boxes_frac": BOX_TOL},
           "what": "tiled YOLO26s (1024 tiles, 6 per 1920x1080 frame) + merge, AIR-MOT val, one GPU",
           "rows": rows}
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{'variant':20s} {'ms/f':>7s} {'x':>5s} {'plane AP50':>10s} {'ship AP50':>10s} parity")
    for r in rows:
        c = r["classes"]
        print(f"{r['variant']:20s} {r['detect_ms_per_frame']:7.2f} {r['speedup_vs_ref']:5.2f} "
              f"{c['plane']['ap50']:10.4f} {c['ship']['ap50']:10.4f} {r['parity_ok']}")


if __name__ == "__main__":
    main()
