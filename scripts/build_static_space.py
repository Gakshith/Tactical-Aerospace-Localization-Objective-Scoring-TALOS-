"""Build the free (static) Hugging Face Space: one index.html + the pre-rendered VISO clips.

    python scripts/build_static_space.py --examples ~/talos/space_examples --out ~/talos/build/space_static

Hugging Face hosts Gradio Spaces only with a PRO plan; static Spaces are free, so this page
shows the pre-rendered examples and the measured tables, with no server. Every number on
the page is read here from a results/ file or the examples' summary.json; nothing is typed.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/Gakshith/Tactical-Aerospace-Localization-Objective-Scoring-TALOS-"


def j(p: Path) -> dict:
    return json.loads(p.read_text())


def e(x) -> str:
    return html.escape(str(x))


def status(r: dict) -> str:
    if r["lead_s"] is not None and r["lead_s"] > 0:
        return "warned"
    if r["lead_s"] is not None:
        return "late"
    return "missed"


def first_alerts(d: Path) -> list[dict]:
    first = {}
    for line in (d / "alerts.jsonl").read_text().splitlines():
        a = json.loads(line)
        first.setdefault((a["class"], a["id"]), a)
    return list(first.values())


def tracking_table(cls: str) -> str:
    s = j(REPO / f"results/final-test-y26s1024-{cls}/metrics.json")["scores"]
    names = [("v1", "TALOS v1"), ("sort", "SORT"), ("bytetrack", "ByteTrack"), ("ocsort", "OC-SORT"),
             ("botsort", "BoT-SORT"), ("talos", "TALOS-Track")]
    best = {m: max(s[k][m] for k, _ in names) for m in ("HOTA", "IDF1", "MOTA")}
    rows = []
    for k, label in names:
        r = s[k]
        cells = "".join(f'<td class="num{" best" if r[m] == best[m] else ""}">{r[m]:.1f}</td>'
                        for m in ("HOTA", "IDF1", "MOTA"))
        rows.append(f'<tr class="{"ours" if k == "talos" else ""}"><th scope="row">{label}</th>{cells}'
                    f'<td class="num">{int(r["IDSW"])}</td></tr>')
    return ('<table><thead><tr><th scope="col">tracker</th><th scope="col" class="num">HOTA</th>'
            '<th scope="col" class="num">IDF1</th><th scope="col" class="num">MOTA</th>'
            '<th scope="col" class="num">ID switches</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def threat_table() -> str:
    rows = []
    for label, f in (("AIR-MOT val", "d4-threat-val-accel1.json"), ("VISO, never trained on", "d4-threat-viso-cross.json")):
        d = j(REPO / "results/threat" / f)
        v2, v1 = d["v2_entry_prob"], d["v1_score"]
        rows.append(f'<tr><th scope="row">{label}</th><td class="num best">{v2["auroc"]:.3f}</td>'
                    f'<td class="num">{v1["auroc"]:.3f}</td><td class="num best">{v2["avg_precision"]:.3f}</td>'
                    f'<td class="num">{v1["avg_precision"]:.3f}</td><td class="num">{e(v2["entries_warned"])}</td>'
                    f'<td class="num">{v2["median_lead_s"]} s</td></tr>')
    return ('<table><thead><tr><th scope="col">data</th><th scope="col" class="num">AUROC v2</th>'
            '<th scope="col" class="num">AUROC v1</th><th scope="col" class="num">AP v2</th>'
            '<th scope="col" class="num">AP v1</th><th scope="col" class="num">entries warned</th>'
            '<th scope="col" class="num">median lead</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def speed_table() -> str:
    d = j(REPO / "results/speed/detector_speed.json")
    nice = {"rt_torch_fp32": "PyTorch fp32", "rt_torch_fp16": "PyTorch fp16", "rt_ort_cuda_fp32": "ONNX Runtime CUDA fp32",
            "rt_ort_trt_fp16": "ONNX Runtime + TensorRT fp16", "rt_trt_fp16": "TensorRT engine fp16"}
    rows = []
    for r in sorted(d["rows"], key=lambda r: r["detect_ms_per_frame"]):
        c = r["classes"]
        rows.append(f'<tr class="{"ours" if r["variant"] == "rt_ort_trt_fp16" else ""}"><th scope="row">'
                    f'{nice.get(r["variant"], r["variant"])}</th><td class="num">{r["detect_ms_per_frame"]:.1f}</td>'
                    f'<td class="num">{r["speedup_vs_ref"]:.2f}×</td><td class="num">{c["plane"]["ap50"]:.3f}</td>'
                    f'<td class="num">{c["ship"]["ap50"]:.3f}</td><td>{"matches" if r["parity_ok"] else "differs"}</td></tr>')
    return ('<table><thead><tr><th scope="col">detector runtime</th><th scope="col" class="num">ms / frame</th>'
            '<th scope="col" class="num">speed-up</th><th scope="col" class="num">plane AP50</th>'
            '<th scope="col" class="num">ship AP50</th><th scope="col">vs PyTorch</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def cpu_rows() -> str:
    out = []
    for tag, label in (("a100/airmot-043", "A100, PyTorch fp32"), ("cpu32/airmot-043", "32-core CPU, PyTorch"),
                       ("cpu32_ort_fp32/airmot-043", "32-core CPU, ONNX Runtime"),
                       ("cpu2_ort_fp32/airmot-043", "2 CPU threads, ONNX Runtime")):
        t = j(REPO / f"results/e2e/{tag}.json")
        m = t["ms_per_frame"]
        out.append(f'<tr><th scope="row">{label}</th><td class="num">{t["fps_end_to_end"]:.2f}</td>'
                   f'<td class="num">{m["detect"]:.1f}</td><td class="num">{m["track"]:.1f}</td>'
                   f'<td class="num">{m["threat"]:.1f}</td></tr>')
    return ('<table><thead><tr><th scope="col">whole pipeline, 1920×1080</th><th scope="col" class="num">fps</th>'
            '<th scope="col" class="num">detect ms</th><th scope="col" class="num">track ms</th>'
            '<th scope="col" class="num">threat ms</th></tr></thead><tbody>' + "".join(out) + "</tbody></table>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = j(a.examples / "summary.json")
    if a.out.exists():
        shutil.rmtree(a.out)
    (a.out / "clips").mkdir(parents=True)

    plane = j(REPO / "results/final-test-y26s1024-plane/metrics.json")["scores"]
    viso = j(REPO / "results/threat/d4-threat-viso-cross.json")
    speed = {r["variant"]: r for r in j(REPO / "results/speed/detector_speed.json")["rows"]}
    n_warned = sum(1 for r in rows if status(r) == "warned")

    clips, buttons = [], []
    for i, r in enumerate(rows):
        shutil.copy(a.examples / r["seq"] / "annotated.mp4", a.out / "clips" / f"{r['seq']}.mp4")
        st = status(r)
        fa = first_alerts(a.examples / r["seq"])
        alert_rows = "".join(f'<tr><td>{x["class"]} {x["id"]}</td><td class="num">{x["frame"]}</td>'
                             f'<td class="num">{x["p_entry"]:.2f}</td><td class="num">'
                             f'{"–" if x["time_to_entry_s"] is None else x["time_to_entry_s"]}</td></tr>' for x in fa)
        alerts = ('<table class="mini"><thead><tr><th>track</th><th class="num">first alert, frame</th>'
                  '<th class="num">P(entry)</th><th class="num">time to entry, s</th></tr></thead><tbody>'
                  + alert_rows + "</tbody></table>") if fa else '<p class="muted">No track crossed the alert threshold.</p>'
        facts = [("labelled objects found", f'{r["recall_iou0.3"]:.0%}'), ("frames", r["frames"]),
                 ("size", f'{r["size"][0]}×{r["size"][1]}')]
        if r["gt_entry_frame"]:
            facts.insert(0, ("enters zone at frame", r["gt_entry_frame"]))
        dl = "".join(f"<div><dt>{e(k)}</dt><dd class=\"num\">{e(v)}</dd></div>" for k, v in facts)
        buttons.append(f'<button class="clip-btn" role="tab" aria-selected="{"true" if i == 0 else "false"}" '
                       f'aria-controls="clip-{i}" id="tab-{i}" data-i="{i}"><span class="dot {st}" aria-hidden="true">'
                       f'</span><span class="seq">{e(r["seq"])}</span><span class="cls">{e(r["class"])}</span>'
                       f'<span class="vd">{e(r["verdict"])}</span></button>')
        clips.append(f'<article class="clip" role="tabpanel" id="clip-{i}" aria-labelledby="tab-{i}"{"" if i == 0 else " hidden"}>'
                     f'<div class="frame"><video src="clips/{e(r["seq"])}.mp4" muted loop playsinline controls '
                     f'preload="{"auto" if i == 0 else "none"}"{" autoplay" if i == 0 else ""}></video></div>'
                     f'<div class="clip-body"><p class="verdict {st}">{e(r["verdict"][0].upper() + r["verdict"][1:])}.</p>'
                     f'<dl class="facts">{dl}</dl>{alerts}</div></article>')

    page = TEMPLATE
    subs = {
        "{{HOTA}}": f'{plane["talos"]["HOTA"]:.1f}', "{{HOTA_BOT}}": f'{plane["botsort"]["HOTA"]:.1f}',
        "{{IDF1}}": f'{plane["talos"]["IDF1"]:.1f}', "{{IDSW}}": str(int(plane["talos"]["IDSW"])),
        "{{AUROC}}": f'{viso["v2_entry_prob"]["auroc"]:.3f}', "{{AUROC_V1}}": f'{viso["v1_score"]["auroc"]:.3f}',
        "{{TRT_MS}}": f'{speed["rt_ort_trt_fp16"]["detect_ms_per_frame"]:.1f}',
        "{{TRT_X}}": f'{speed["rt_ort_trt_fp16"]["speedup_vs_ref"]:.1f}',
        "{{N_WARNED}}": str(n_warned), "{{N_CLIPS}}": str(len(rows)),
        "{{BUTTONS}}": "".join(buttons), "{{CLIPS}}": "".join(clips),
        "{{T_PLANE}}": tracking_table("plane"), "{{T_SHIP}}": tracking_table("ship"),
        "{{T_THREAT}}": threat_table(), "{{T_SPEED}}": speed_table(), "{{T_E2E}}": cpu_rows(), "{{GITHUB}}": GITHUB,
    }
    for k, v in subs.items():
        page = page.replace(k, v)
    assert "{{" not in page, "unfilled placeholder"
    (a.out / "index.html").write_text(page)
    (a.out / "README.md").write_text(README)
    (a.out / ".gitattributes").write_text("*.mp4 filter=lfs diff=lfs merge=lfs -text\n")
    print(f"wrote {a.out} ({len(rows)} clips, {n_warned} warned)")


README = """---
title: TALOS Satellite Tracking
emoji: 🛰️
colorFrom: gray
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: cc-by-nc-sa-4.0
short_description: Aircraft tracking and zone-entry warnings in satellite video
---

TALOS: aircraft and ship tracking with zone-entry warnings in satellite video. The example clips are
from VISO (CC BY-NC-SA 4.0, Yin et al., IEEE TGRS 2021) and are shared under the same license. The
code is MIT: https://github.com/Gakshith/Tactical-Aerospace-Localization-Objective-Scoring-TALOS-
"""

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TALOS · satellite tracking</title>
<meta name="description" content="Aircraft tracking and zone-entry warnings in satellite video, measured against SORT, ByteTrack, OC-SORT and BoT-SORT.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#f6f5f1; --card:#fffefb; --ink:#15171b; --ink-2:#4a4e55; --muted:#7a7e85; --line:#e2dfd8; --line-2:#ecebe6;
  --alert:#c2362b; --alert-tint:#f6e3df; --late:#b7791f; --late-tint:#f5ead6; --miss:#9a9da3;
  --radius:10px; --shell:14px;
  --sans:"Instrument Sans",ui-sans-serif,system-ui,sans-serif; --mono:"JetBrains Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#121315; --card:#191a1d; --ink:#ecebe7; --ink-2:#b9b8b3; --muted:#8b8a86; --line:#2c2d31; --line-2:#232427;
  --alert:#e5604f; --alert-tint:#3a1f1b; --late:#d9a24a; --late-tint:#33291a; --miss:#6f7176;}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:400 16px/1.55 var(--sans);letter-spacing:-.005em}
a{color:inherit;text-decoration-thickness:1px;text-underline-offset:3px}
a:hover{color:var(--alert)}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px}
.label{font:500 11.5px/1 var(--mono);text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.num{font-variant-numeric:tabular-nums}
/* signature: the protected-zone ring */
.ring{display:inline-block;width:14px;height:14px;border:1.5px solid var(--alert);border-radius:50%;vertical-align:-2px;margin-right:10px}
header.top{display:flex;justify-content:space-between;align-items:center;padding:22px 0;border-bottom:1px solid var(--line)}
.brand{font:500 13px/1 var(--mono);letter-spacing:.14em}
.top nav{display:flex;gap:22px;font-size:14px;color:var(--ink-2)}
.top nav a{text-decoration:none}
.hero{padding:72px 0 40px;display:grid;grid-template-columns:1.25fr 1fr;gap:48px;align-items:end}
h1{font-weight:400;font-size:clamp(34px,5.2vw,62px);line-height:1.04;letter-spacing:-.035em;margin:18px 0 0}
h1 em{font-style:normal;color:var(--alert)}
.lede{color:var(--ink-2);font-size:17px;max-width:46ch;margin:0}
.stats{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-radius:var(--radius);background:var(--card);margin:8px 0 0}
.stat{padding:20px 20px 18px;border-left:1px solid var(--line)}
.stat:first-child{border-left:0}
.stat b{display:block;font:400 34px/1 var(--sans);letter-spacing:-.03em;font-variant-numeric:tabular-nums;margin:12px 0 8px}
.stat span{font-size:13.5px;color:var(--ink-2);line-height:1.4;display:block}
section{padding:72px 0 8px}
.sec-head{display:grid;grid-template-columns:1fr 2fr;gap:48px;margin-bottom:28px;align-items:start}
h2{font-weight:400;font-size:clamp(26px,3vw,36px);line-height:1.1;letter-spacing:-.025em;margin:12px 0 0}
.sec-head p{margin:0;color:var(--ink-2);max-width:62ch}
/* shell + inset card */
.shell{background:var(--line-2);border:1px solid var(--line);border-radius:var(--shell);padding:6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius)}
.player{display:grid;grid-template-columns:280px 1fr;gap:6px}
.list{display:flex;flex-direction:column;padding:6px}
.clip-btn{all:unset;cursor:pointer;display:grid;grid-template-columns:14px auto 1fr;grid-template-rows:auto auto;column-gap:10px;row-gap:3px;
  padding:11px 12px;border-radius:8px;border:1px solid transparent;transition:background .15s,border-color .15s}
.clip-btn:hover{background:var(--bg)}
.clip-btn[aria-selected="true"]{background:var(--bg);border-color:var(--line)}
.clip-btn:focus-visible{outline:2px solid var(--alert);outline-offset:1px}
.dot{width:9px;height:9px;border-radius:50%;margin-top:6px;grid-row:span 2}
.dot.warned{background:var(--alert)} .dot.late{background:var(--late)} .dot.missed{background:transparent;border:1.5px solid var(--miss)}
.seq{font:500 13.5px/1.4 var(--mono)}
.cls{font:400 12px/1.6 var(--mono);color:var(--muted);text-align:right}
.vd{grid-column:2/4;font-size:13px;color:var(--ink-2);line-height:1.35}
.clip{display:grid;grid-template-rows:auto auto}
.clip[hidden]{display:none}
.player>.card{min-width:0}
.frame{background:#0d0e10;border-radius:var(--radius) var(--radius) 0 0;display:flex;align-items:center;justify-content:center;aspect-ratio:16/9;width:100%;overflow:hidden}
@media (min-width:901px){.frame{min-height:300px}}
.frame video{max-width:100%;max-height:100%;display:block}
.clip-body{padding:20px 22px 22px;display:grid;grid-template-columns:1fr 1.2fr;gap:28px;align-items:start}
.verdict{margin:0 0 14px;font-size:20px;letter-spacing:-.015em;line-height:1.25}
.verdict.warned{color:var(--alert)} .verdict.late{color:var(--late)} .verdict.missed{color:var(--ink-2)}
.facts{display:grid;grid-template-columns:1fr 1fr;gap:12px 20px;margin:0}
.facts div{border-top:1px solid var(--line);padding-top:8px}
.facts dt{font:500 11px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.facts dd{margin:4px 0 0;font-size:17px}
.note{font-size:13.5px;color:var(--muted);margin:14px 0 0;max-width:80ch}
table{width:100%;border-collapse:collapse;font-size:14.5px}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid var(--line-2);white-space:nowrap}
thead th{font:500 11px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.07em;color:var(--muted);border-bottom:1px solid var(--line)}
tbody th{font-weight:400}
td.num,th.num{text-align:right}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}
tr.ours th{font-weight:600}
tr.ours th::before{content:"";display:inline-block;width:8px;height:8px;border:1.5px solid var(--alert);border-radius:50%;margin-right:8px;vertical-align:0}
td.best{color:var(--alert);font-weight:500}
table.mini{font-size:13.5px} table.mini th,table.mini td{padding:7px 10px}
.tables{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.tables .card{overflow-x:auto}
.cap{padding:16px 16px 6px}
.cap h3{font-weight:500;font-size:16px;letter-spacing:-.01em;margin:8px 0 2px}
.cap p{margin:0;color:var(--ink-2);font-size:13.5px}
.span2{grid-column:1/-1}
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
.step{padding:20px}
.step h3{font-weight:500;font-size:17px;margin:12px 0 8px;letter-spacing:-.01em}
.step p{margin:0;font-size:14.5px;color:var(--ink-2)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:48px}
.cols ul{margin:12px 0 0;padding-left:18px;color:var(--ink-2)} .cols li{margin:6px 0}
pre{font:400 13px/1.6 var(--mono);background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:16px;overflow-x:auto;margin:12px 0 0}
footer{margin-top:80px;border-top:1px solid var(--line);padding:28px 0 48px;display:flex;justify-content:space-between;gap:24px;color:var(--muted);font-size:13.5px;flex-wrap:wrap}
@media (max-width:900px){
  .hero{grid-template-columns:1fr;gap:20px;padding-top:44px}
  .stats{grid-template-columns:1fr 1fr} .stat:nth-child(3){border-left:0} .stat:nth-child(n+3){border-top:1px solid var(--line)}
  .sec-head,.cols{grid-template-columns:1fr;gap:12px}
  .player{grid-template-columns:1fr} .list{flex-direction:row;overflow-x:auto;gap:4px}
  .clip-btn{min-width:210px}
  .clip-body,.tables,.steps{grid-template-columns:1fr}
  .top nav{display:none}
}
@media (max-width:520px){.wrap{padding:0 16px} .stats{grid-template-columns:1fr} .stat{border-left:0!important;border-top:1px solid var(--line)} .stat:first-child{border-top:0}}
@media (prefers-reduced-motion:no-preference){
  .rise{opacity:0;transform:translateY(8px);animation:rise .7s cubic-bezier(.16,1,.3,1) forwards}
  .rise.d1{animation-delay:.06s}.rise.d2{animation-delay:.12s}.rise.d3{animation-delay:.18s}
  @keyframes rise{to{opacity:1;transform:none}}
}
</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <div class="brand"><span class="ring" aria-hidden="true"></span>TALOS</div>
  <nav aria-label="Sections"><a href="#examples">Examples</a><a href="#results">Results</a><a href="#how">How it works</a><a href="{{GITHUB}}">Code</a></nav>
</header>

<main>
<section class="hero" style="padding-top:64px">
  <div class="rise">
    <div class="label">Satellite video · tracking · zone warnings</div>
    <h1>Warns before an aircraft <em>reaches the zone</em>, from satellite video.</h1>
  </div>
  <p class="lede rise d1">TALOS detects aircraft and ships in Jilin-1 satellite video, tracks each one, and estimates the probability that it enters a protected zone in the next 10 seconds. Every number below is measured on held-out video and read from the run's result file.</p>
</section>
<div class="stats rise d2" role="list">
  <div class="stat" role="listitem"><div class="label">Tracking, planes</div><b>{{HOTA}}</b><span>HOTA on AIR-MOT test, vs {{HOTA_BOT}} for BoT-SORT; all trackers tuned with the same budget</span></div>
  <div class="stat" role="listitem"><div class="label">Identity</div><b>{{IDF1}}</b><span>IDF1 with {{IDSW}} ID switches on the same test set</span></div>
  <div class="stat" role="listitem"><div class="label">Threat ranking</div><b>{{AUROC}}</b><span>AUROC on VISO, a dataset never trained on (v1 score: {{AUROC_V1}})</span></div>
  <div class="stat" role="listitem"><div class="label">Detector speed</div><b>{{TRT_MS}} ms</b><span>per 1920×1080 frame on an A100 with TensorRT fp16, {{TRT_X}}× PyTorch, same accuracy</span></div>
</div>

<section id="examples" aria-labelledby="ex-h">
  <div class="sec-head">
    <div><div class="label"><span class="ring" aria-hidden="true"></span>01 · Examples</div><h2 id="ex-h">Eight clips it has never seen</h2></div>
    <p>Every VISO clip with a labelled aircraft or ship, failures included. A circle zone (white in the video) is placed on a labelled object's path, so you can watch whether the warning comes in time. Boxes turn <strong style="color:var(--alert);font-weight:500">red</strong> when entry within 10 s becomes likely. It warned in time on <strong>{{N_WARNED}} of {{N_CLIPS}}</strong>.</p>
  </div>
  <div class="shell player">
    <div class="card list" role="tablist" aria-label="Example clips">{{BUTTONS}}</div>
    <div class="card">{{CLIPS}}</div>
  </div>
  <p class="note">Lead time counts only alerts on the track that follows the labelled object. VISO labels only moving objects, so parked aircraft show up as extra boxes. The detector was trained on AIR-MOT and misses VISO's ships, which are a few pixels across.</p>
</section>

<section id="results" aria-labelledby="res-h">
  <div class="sec-head">
    <div><div class="label"><span class="ring" aria-hidden="true"></span>02 · Results</div><h2 id="res-h">Measured against published trackers</h2></div>
    <p>Test runs once per final config. All five trackers get the same detections and the same 30-config tuning budget on val. Best value per column in red. Where a baseline wins, the table shows it: on ships, BoT-SORT has the higher MOTA.</p>
  </div>
  <div class="shell tables">
    <div class="card"><div class="cap"><div class="label">Tracking · AIR-MOT test</div><h3>Aircraft</h3></div>{{T_PLANE}}</div>
    <div class="card"><div class="cap"><div class="label">Tracking · AIR-MOT test</div><h3>Ships</h3></div>{{T_SHIP}}</div>
    <div class="card span2"><div class="cap"><div class="label">Threat</div><h3>Does P(enter zone within 10 s) rank the zones that really get entered?</h3><p>v2 is the filter-covariance probability; v1 is the original hand-weighted score. Entries warned at a 5% false-alarm rate.</p></div>{{T_THREAT}}</div>
    <div class="card span2"><div class="cap"><div class="label">Speed · AIR-MOT val, one A100</div><h3>Detector runtimes</h3><p>"Matches" = AP50 within 0.005 and box count within 1% of PyTorch fp32.</p></div>{{T_SPEED}}</div>
    <div class="card span2"><div class="cap"><div class="label">Speed · end to end</div><h3>Video in, alerts out</h3></div>{{T_E2E}}</div>
  </div>
</section>

<section id="how" aria-labelledby="how-h">
  <div class="sec-head">
    <div><div class="label"><span class="ring" aria-hidden="true"></span>03 · How it works</div><h2 id="how-h">Four stages, each timed</h2></div>
    <p>Detection is nearly all the cost; tracking and threat scoring together take under 2 ms per frame on the A100.</p>
  </div>
  <div class="shell steps">
    <div class="card step"><div class="label">Detect</div><h3>Tiled YOLO26s</h3><p>1024 px tiles keep 10–30 px aircraft at native resolution. An edge-aware merge rejoins objects cut by tile borders.</p></div>
    <div class="card step"><div class="label">Track</div><h3>TALOS-Track</h3><p>A Kalman filter per object with Mahalanobis gating, two-stage association and duplicate suppression. Measurement noise is calibrated from detector errors.</p></div>
    <div class="card step"><div class="label">Threat</div><h3>P(entry within 10 s)</h3><p>Monte Carlo rollouts from each track's state and covariance. The alert threshold is the 5% false-alarm point on val.</p></div>
    <div class="card step"><div class="label">Deploy</div><h3>ONNX and TensorRT</h3><p>The exported model matches PyTorch accuracy and runs without PyTorch on CPU, or 2.1× faster with TensorRT.</p></div>
  </div>
</section>

<section aria-labelledby="lim-h">
  <div class="cols">
    <div><div class="label"><span class="ring" aria-hidden="true"></span>04 · Limits</div><h2 id="lim-h">What it does not do yet</h2>
      <ul><li>Aircraft under 16 px are mostly missed (val recall 5%).</li><li>Ships carry many false positives (val precision 0.32).</li><li>Distances are in pixels. AIR-MOT does not state its frame rate; 10 fps is assumed.</li><li>This page is pre-rendered; free Spaces no longer run Python apps.</li></ul></div>
    <div><div class="label">Run it yourself</div><h2>From the repo</h2>
<pre>python -m src.TALOS.pipeline.run VIDEO_OR_FRAME_DIR \
  --out out/ --zone-circle CX,CY,R --device cpu \
  --backend onnx --weights talos_y26s_t1024.onnx</pre></div>
  </div>
</section>
</main>

<footer>
  <div>Example clips: VISO, CC BY-NC-SA 4.0 (Yin et al., IEEE TGRS 2021), shared under the same license. Detector trained on AIR-MOT (He et al., IEEE TGRS 2022); no AIR-MOT frames shown.</div>
  <div><a href="{{GITHUB}}">Code (MIT)</a></div>
</footer>
</div>
<script>
const tabs=[...document.querySelectorAll('.clip-btn')], panels=[...document.querySelectorAll('.clip')];
function show(i){tabs.forEach((t,k)=>t.setAttribute('aria-selected',k===i));
  panels.forEach((p,k)=>{const v=p.querySelector('video');if(k===i){p.hidden=false;v.preload='auto';v.play().catch(()=>{});}else{p.hidden=true;v.pause();}});}
tabs.forEach((t,i)=>{t.addEventListener('click',()=>show(i));
  t.addEventListener('keydown',ev=>{const d={ArrowDown:1,ArrowRight:1,ArrowUp:-1,ArrowLeft:-1}[ev.key];
    if(d){ev.preventDefault();const n=(i+d+tabs.length)%tabs.length;tabs[n].focus();show(n);}});});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
