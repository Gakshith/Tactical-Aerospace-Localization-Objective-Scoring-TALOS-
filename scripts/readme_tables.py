"""Regenerate every results table in README.md from results/ (never typed by hand).

    python scripts/readme_tables.py            # rewrite README.md in place
    python scripts/readme_tables.py --check    # exit 1 if README.md is out of date

Each table sits between ``<!-- table:NAME -->`` and ``<!-- /table:NAME -->``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
NAMES = [("v1", "TALOS v1"), ("sort", "SORT"), ("bytetrack", "ByteTrack"), ("ocsort", "OC-SORT"),
         ("botsort", "BoT-SORT"), ("talos", "**TALOS-Track**")]


def j(p) -> dict:
    return json.loads((R / p).read_text())


def md(header: list[str], rows: list[list], right_from: int = 1) -> str:
    al = ["---" if i < right_from else "---:" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(al) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def bold_best(vals: list[float], fmt="{:.1f}") -> list[str]:
    best = max(vals)
    return [f"**{fmt.format(v)}**" if v == best else fmt.format(v) for v in vals]


def t_tracking() -> str:
    out = []
    for cls in ("plane", "ship"):
        s = j(f"final-test-y26s1024-{cls}/metrics.json")["scores"]
        cols = {m: bold_best([s[k][m] for k, _ in NAMES]) for m in ("HOTA", "IDF1", "MOTA", "AssA")}
        rows = [[label, cols["HOTA"][i], cols["IDF1"][i], cols["MOTA"][i], cols["AssA"][i], int(s[k]["IDSW"])]
                for i, (k, label) in enumerate(NAMES)]
        out.append(f"**{'Aircraft' if cls == 'plane' else 'Ships'}** (AIR-MOT test, {len(j(f'final-test-y26s1024-{cls}/metrics.json')['config']['seqs'])} videos)\n\n"
                   + md(["tracker", "HOTA", "IDF1", "MOTA", "AssA", "ID switches"], rows))
    before = j("d2-before-v1-plane-test/metrics.json")["scores"]["v1"]
    out.append(f"The whole v1 system (its own detector and tracker) scores **{before['HOTA']:.1f} HOTA** on the "
               "same aircraft test videos: its detector, trained on synthetic images, finds almost nothing in real satellite video.")
    return "\n\n".join(out)


def t_tuning() -> str:
    """Default vs tuned val HOTA for each tracker, with both detectors that were tuned."""
    rows = []
    for cls in ("plane", "ship"):
        final = j(f"d4-tune-y26s1024-{cls}-val/tuning.json")["best"]
        final["talos"] = j(f"d4-tune-talos-dedup-{cls}-val/tuning.json")["best"]["talos"]
        early = j(f"d4-tune-y26m-{cls}-val/tuning.json")["best"]
        for k, label in NAMES[1:]:
            f, e_ = final[k], early[k]
            rows.append([f"{label} ({cls}s)",
                         f"{f['default_scores']['HOTA']:.1f} → {f['scores']['HOTA']:.1f}",
                         f"{f['scores']['HOTA'] - f['default_scores']['HOTA']:+.1f}",
                         f"{e_['default_scores']['HOTA']:.1f} → {e_['scores']['HOTA']:.1f}",
                         f"{e_['scores']['HOTA'] - e_['default_scores']['HOTA']:+.1f}"])
    return md(["val HOTA, defaults → tuned", "final detector (YOLO26s, 1024 tiles)", "gain",
               "earlier detector (YOLO26m, 640 tiles)", "gain"], rows)


def t_dedup() -> str:
    rows = []
    for cls in ("plane", "ship"):
        pre = j(f"d4-tune-y26s1024-{cls}-val/tuning.json")["best"]["talos"]["scores"]
        post = j(f"d4-tune-talos-dedup-{cls}-val/tuning.json")["best"]["talos"]["scores"]
        rows.append([f"{cls}s", int(pre["IDSW"]), int(post["IDSW"]), f"{pre['HOTA']:.1f} → {post['HOTA']:.1f}",
                     f"{pre['IDF1']:.1f} → {post['IDF1']:.1f}"])
    return md(["val", "ID switches before", "after", "HOTA", "IDF1"], rows)


def t_ablation() -> str:
    s = j("d3-ablation-calib-noisy-plane-val/metrics.json")["scores"]
    steps = [("talos:a0-base", "constant-velocity Kalman + Hungarian (v1-like)"),
             ("talos:a1-lifecycle", "+ track lifecycle (tentative / confirmed / lost)"),
             ("talos:a2-real-dt", "+ real time step"), ("talos:a3-maha-gate", "+ Mahalanobis gate"),
             ("talos:a4-two-stage", "+ two-stage (ByteTrack-style) association"),
             ("talos:a5-reupdate", "+ OC-SORT re-update"), ("talos:a6-coast-out", "+ coasting output")]
    rows = [[lab, f"{s[k]['HOTA']:.1f}", f"{s[k]['IDF1']:.1f}", f"{s[k]['MOTA']:.1f}", int(s[k]["IDSW"])] for k, lab in steps]
    b = s["bytetrack"]
    rows.append(["ByteTrack, same detections", f"{b['HOTA']:.1f}", f"{b['IDF1']:.1f}", f"{b['MOTA']:.1f}", int(b["IDSW"])])
    return md(["cumulative step", "HOTA", "IDF1", "MOTA", "ID switches"], rows)


def t_detector() -> str:
    rows = []
    v1 = j("det_v1_cgi/metrics.json")["AIRMOT_plane"]["splits"]
    for sp in ("val", "test"):
        for cls in ("plane", "ship"):
            f = f"det_y26s_t1024/{cls}.json" if sp == "val" else f"det_y26s_t1024/{cls}-test.json"
            r = j(f)[f"AIRMOT_{cls}"]["splits"][sp]
            rb = r["recall_by_size"]
            at = next(v for k, v in r.items() if k.startswith("at_conf_"))
            rows.append([f"{cls}s, {sp}", f"{r['ap50']:.3f}", f"{at['precision']:.2f}", f"{at['recall']:.2f}",
                         *(f"{rb[b]['recall_at_0.25']:.2f}" if rb[b]["gt_boxes"] else "–" for b in ("lt16", "16to32", "gt32"))])
    rows.append(["v1 detector, planes, val", f"{v1['val']['ap50']:.3f}", "", "", "", "", ""])
    return md(["AIR-MOT", "AP50", "precision", "recall", "recall <16 px", "16–32 px", ">32 px"], rows)


def t_threat() -> str:
    rows = []
    for name, f in (("AIR-MOT val", "threat/d4-threat-val-accel1.json"), ("VISO (never trained on)", "threat/d4-threat-viso-cross.json")):
        d = j(f)
        v2, v1 = d["v2_entry_prob"], d["v1_score"]
        rows.append([name, d["zones"]["on-path"] + d["zones"]["near-miss"], f"**{v2['auroc']:.3f}**", f"{v1['auroc']:.3f}",
                     f"**{v2['avg_precision']:.3f}**", f"{v1['avg_precision']:.3f}", v2["entries_warned"], v1["entries_warned"],
                     f"{v2['median_lead_s']} s"])
    return md(["data", "zones", "AUROC v2", "AUROC v1", "AP v2", "AP v1", "warned v2", "warned v1", "median lead v2"], rows)


def t_speed() -> str:
    d = j("speed/detector_speed.json")
    nice = {"rt_torch_fp32": "PyTorch fp32 (reference)", "rt_torch_fp16": "PyTorch fp16",
            "rt_ort_cuda_fp32": "ONNX Runtime, CUDA fp32", "rt_ort_trt_fp16": "**ONNX Runtime + TensorRT fp16**",
            "rt_trt_fp16": "TensorRT engine (Ultralytics) fp16"}
    rows = [[nice[r["variant"]], f"{r['detect_ms_per_frame']:.1f}", f"{r['speedup_vs_ref']:.2f}×",
             f"{r['classes']['plane']['ap50']:.4f}", f"{r['classes']['ship']['ap50']:.4f}", "yes" if r["parity_ok"] else "no"]
            for r in sorted(d["rows"], key=lambda r: r["detect_ms_per_frame"])]
    return md(["runtime (one A100)", "ms / frame", "speed-up", "plane AP50", "ship AP50", "matches PyTorch"], rows)


def t_e2e() -> str:
    rows = []
    for tag, label in (("a100/airmot-043", "A100, PyTorch fp32"), ("cpu32/airmot-043", "32 CPU cores, PyTorch"),
                       ("cpu32_ort_fp32/airmot-043", "32 CPU cores, ONNX Runtime fp32"),
                       ("cpu32_ort_int8dyn/airmot-043", "32 CPU cores, ONNX Runtime dynamic INT8"),
                       ("cpu2_ort_fp32/airmot-043", "2 CPU threads, ONNX Runtime fp32")):
        t = j(f"e2e/{tag}.json")
        m = t["ms_per_frame"]
        rows.append([label, t["frames"], f"{t['fps_end_to_end']:.2f}", f"{m['decode']:.1f}", f"{m['detect']:.1f}",
                     f"{m['track']:.1f}", f"{m['threat']:.1f}"])
    return md(["hardware, runtime", "frames", "fps", "decode ms", "detect ms", "track ms", "threat ms"], rows)


def t_viso_clips() -> str:
    p = ROOT / "results" / "space" / "summary.json"
    rows = [[r["seq"], r["class"], f"{r['recall_iou0.3']:.0%}", r["verdict"]] for r in json.loads(p.read_text())]
    return md(["VISO clip", "class", "labelled objects found", "zone warning"], rows, right_from=4)


TABLES = {"tracking": t_tracking, "tuning": t_tuning, "dedup": t_dedup, "ablation": t_ablation,
          "detector": t_detector, "threat": t_threat, "speed": t_speed, "e2e": t_e2e, "viso": t_viso_clips}


def render(text: str) -> str:
    for name, fn in TABLES.items():
        pat = re.compile(rf"(<!-- table:{name} -->\n)(?:.*?\n)?(<!-- /table:{name} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"README has no markers for table '{name}'")
        text = pat.sub(lambda m: m.group(1) + fn() + "\n" + m.group(2), text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    p = ROOT / "README.md"
    old = p.read_text()
    new = render(old)
    if a.check:
        sys.exit(0 if new == old else 1)
    p.write_text(new)
    print("README.md tables regenerated")


if __name__ == "__main__":
    main()
