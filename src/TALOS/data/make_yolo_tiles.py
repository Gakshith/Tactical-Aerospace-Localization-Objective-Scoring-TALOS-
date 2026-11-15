"""Build a tiled YOLO detection dataset from the converted AIR-MOT benchmarks.

Objects are tiny (aircraft median 34 px, ships 10 px, in 1920x1080 frames), so
we train on native-resolution square tiles instead of shrinking whole frames.

Layout written to ``<out-root>/<name>/``::

    images/{train,val}/<seq>_f<frame>_x<x0>_y<y0>.jpg
    labels/{train,val}/<same stem>.txt    # "cls cx cy w h", normalised to the tile
    data.yaml                              # names {0: plane, 1: ship}
    stats.json                             # counts, box sizes, settings

Rules:
  * splits come from ``configs/splits/airmot.json``; only train and val are
    read, never test
  * every ``--stride-<split>``-th frame is used (consecutive frames are
    near-duplicates); frame 1 is always first
  * tiles of ``--tile`` px overlap by ``--overlap``; the last tile in each axis
    sits flush with the frame edge
  * a box is labelled in a tile if >= ``--min-inside`` of its area is inside,
    and is then clipped to the tile
  * tiles that are >= ``--max-black`` black (bottom bands, registration
    margins, the black tail of video 48) are dropped
  * every tile with a label is kept; label-free tiles are sampled at
    ``--neg-ratio`` x the positive count (fixed seed)

A video can be in both benchmarks (50, 52): its plane and ship boxes are merged
onto one set of frames.

    python -m src.TALOS.data.make_yolo_tiles --name airmot_t640 --tile 640
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.TALOS.data.mot import read_mot, read_seqinfo

CLASSES = {0: "plane", 1: "ship"}
BENCHES = {0: "AIRMOT_plane-all", 1: "AIRMOT_ship-all"}
BLACK_LEVEL = 10  # a pixel is "black" if every channel is <= this (JPEG noise)


# ---------------------------------------------------------------- pure logic
def tile_starts(length: int, tile: int, overlap: float) -> list[int]:
    """Start offsets covering [0, length) with tiles of ``tile`` px.

    Step is ``tile * (1 - overlap)``; the last tile is flush with the edge.
    If the frame is smaller than a tile, one tile starts at 0 (it is cropped
    to the frame)."""
    if length <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    starts = list(range(0, length - tile, step))
    starts.append(length - tile)
    return sorted(set(starts))


def tile_grid(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """All tiles as (x0, y0, x1, y1), row-major."""
    xs, ys = tile_starts(width, tile, overlap), tile_starts(height, tile, overlap)
    return [(x, y, min(x + tile, width), min(y + tile, height)) for y in ys for x in xs]


def boxes_in_tile(
    boxes_xyxy: np.ndarray, cls: np.ndarray, tile_xyxy: tuple[int, int, int, int], min_inside: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Boxes with >= ``min_inside`` of their area inside the tile, clipped to it
    and shifted to tile coordinates. Returns (xyxy in tile px, cls)."""
    b = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    c = np.asarray(cls).reshape(-1)
    if b.shape[0] == 0:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    tx0, ty0, tx1, ty1 = tile_xyxy
    cx0, cy0 = np.maximum(b[:, 0], tx0), np.maximum(b[:, 1], ty0)
    cx1, cy1 = np.minimum(b[:, 2], tx1), np.minimum(b[:, 3], ty1)
    inter = np.clip(cx1 - cx0, 0, None) * np.clip(cy1 - cy0, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep = (area > 0) & (inter >= min_inside * area) & (inter > 0)
    out = np.column_stack([cx0 - tx0, cy0 - ty0, cx1 - tx0, cy1 - ty0])[keep]
    return out, c[keep].astype(int)


def yolo_lines(boxes_xyxy: np.ndarray, cls: np.ndarray, tile_w: int, tile_h: int) -> list[str]:
    """YOLO label lines (cls cx cy w h, normalised) for boxes in tile px."""
    lines = []
    for (x0, y0, x1, y1), k in zip(np.asarray(boxes_xyxy).reshape(-1, 4), np.asarray(cls).reshape(-1)):
        cx, cy = (x0 + x1) / 2 / tile_w, (y0 + y1) / 2 / tile_h
        w, h = (x1 - x0) / tile_w, (y1 - y0) / tile_h
        lines.append(f"{int(k)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def black_fraction(img: np.ndarray, level: int = BLACK_LEVEL) -> float:
    """Fraction of pixels whose every channel is <= ``level``."""
    if img.size == 0:
        return 1.0
    a = img if img.ndim == 2 else img.max(axis=2)
    return float((a <= level).mean())


def sampled_frames(n_frames: int, stride: int) -> list[int]:
    """1-based frame numbers 1, 1+stride, ..."""
    return list(range(1, n_frames + 1, max(1, stride)))


# ---------------------------------------------------------------- per video
@dataclass
class Video:
    seq: str
    frames: list[str]  # absolute image paths, index = frame - 1
    width: int
    height: int
    boxes: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # frame -> (xyxy, cls)


def load_video(mot_root: Path, seq: str) -> Video | None:
    """Merge plane + ship gt for one video. None if it is in neither benchmark."""
    per_frame: dict[int, list] = defaultdict(list)
    video = None
    for k, bench in BENCHES.items():
        d = mot_root / bench / seq
        if not (d / "seqinfo.ini").exists():
            continue
        info = read_seqinfo(d)
        img_dir = d / info.img_dir
        frames = sorted(str(p) for p in img_dir.iterdir() if p.name.endswith(info.img_ext))
        if len(frames) != info.length:
            raise RuntimeError(f"{d}: {len(frames)} images with {info.img_ext}, seqinfo says {info.length}")
        if video is None:
            video = Video(seq, frames, info.width, info.height)
        elif [Path(f).name for f in frames] != [Path(f).name for f in video.frames]:
            raise RuntimeError(f"{seq}: plane and ship benchmarks list different frames")
        rows = read_mot(d / "gt" / "gt.txt")
        rows = np.unique(rows[:, :6], axis=0)  # v12 has exact duplicate rows
        for r in rows:
            f = int(r[0])
            per_frame[f].append((r[2], r[3], r[2] + r[4], r[3] + r[5], k))
    if video is None:
        return None
    for f, lst in per_frame.items():
        a = np.array(lst, dtype=float)
        video.boxes[f] = (a[:, :4], a[:, 4].astype(int))
    return video


def _read(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return img


def scan_video(job: dict) -> list[dict]:
    """Pass 1: for every sampled frame and tile, its labels and black fraction."""
    v: Video = job["video"]
    out = []
    grid = tile_grid(v.width, v.height, job["tile"], job["overlap"])
    for f in sampled_frames(len(v.frames), job["stride"]):
        img = _read(v.frames[f - 1])
        if img.shape[:2] != (v.height, v.width):
            raise RuntimeError(f"{v.frames[f - 1]}: shape {img.shape}, expected {v.height}x{v.width}")
        b, c = v.boxes.get(f, (np.zeros((0, 4)), np.zeros((0,), int)))
        for t in grid:
            tb, tc = boxes_in_tile(b, c, t, job["min_inside"])
            out.append({
                "seq": v.seq, "frame": f, "tile": t,
                "black": black_fraction(img[t[1]:t[3], t[0]:t[2]]),
                "boxes": tb, "cls": tc,
            })
    return out


def write_video(job: dict) -> int:
    """Pass 2: write the selected tiles of one video."""
    import cv2

    v: Video = job["video"]
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for rec in job["tiles"]:
        by_frame[rec["frame"]].append(rec)
    n = 0
    for f, recs in sorted(by_frame.items()):
        img = _read(v.frames[f - 1])
        for rec in recs:
            x0, y0, x1, y1 = rec["tile"]
            stem = f"{v.seq}_f{f:06d}_x{x0:04d}_y{y0:04d}"
            cv2.imwrite(str(job["img_dir"] / f"{stem}.jpg"), img[y0:y1, x0:x1],
                        [cv2.IMWRITE_JPEG_QUALITY, job["jpeg_quality"]])
            lines = yolo_lines(rec["boxes"], rec["cls"], x1 - x0, y1 - y0)
            (job["lbl_dir"] / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n += 1
    return n


# ---------------------------------------------------------------- stats
def _pct(a: np.ndarray) -> dict:
    if a.size == 0:
        return {}
    q = np.percentile(a, [0, 10, 50, 90, 100])
    return {"min": round(float(q[0]), 1), "p10": round(float(q[1]), 1), "median": round(float(q[2]), 1),
            "p90": round(float(q[3]), 1), "max": round(float(q[4]), 1)}


def split_stats(scanned: list[dict], kept: list[dict], n_frames: int, n_videos: int, n_black: dict) -> dict:
    pos = [r for r in kept if len(r["cls"])]
    s = {
        "videos": n_videos, "frames_sampled": n_frames,
        "tiles_scanned": len(scanned), "tiles_dropped_black": n_black,
        "tiles": len(kept), "tiles_positive": len(pos), "tiles_negative": len(kept) - len(pos),
        "boxes": {}, "box_size_px": {}, "tiles_with_class": {},
    }
    for k, name in CLASSES.items():
        wh = [r["boxes"][r["cls"] == k] for r in kept]
        wh = np.concatenate(wh) if wh else np.zeros((0, 4))
        w, h = wh[:, 2] - wh[:, 0], wh[:, 3] - wh[:, 1]
        side = np.sqrt(w * h)
        s["boxes"][name] = int(len(wh))
        s["tiles_with_class"][name] = int(sum(bool((r["cls"] == k).any()) for r in kept))
        s["box_size_px"][name] = {
            "w": _pct(w), "h": _pct(h), "sqrt_area": _pct(side),
            "lt8px": int((side < 8).sum()), "lt16px": int((side < 16).sum()), "lt32px": int((side < 32).sum()),
        }
    return s


# ---------------------------------------------------------------- main
def build(a: argparse.Namespace) -> dict:
    split_cfg = json.loads(Path(a.split_file).read_text())["split"]
    out = Path(a.out_root).expanduser() / a.name
    mot_root = Path(a.mot_root).expanduser()
    rng = random.Random(a.seed)
    if (out / "images").exists() or (out / "labels").exists():
        if not a.overwrite:
            raise SystemExit(f"{out} already has images/labels; pass --overwrite to rebuild")
        import shutil

        for sub in ("images", "labels"):
            shutil.rmtree(out / sub, ignore_errors=True)
    stats = {
        "name": a.name, "tile": a.tile, "overlap": a.overlap, "min_inside": a.min_inside,
        "max_black": a.max_black, "black_level": BLACK_LEVEL, "neg_ratio": a.neg_ratio, "seed": a.seed,
        "stride": {"train": a.stride_train, "val": a.stride_val}, "names": CLASSES,
        "split_file": a.split_file, "splits": {},
    }
    for split in ("train", "val"):  # never test
        stride = a.stride_train if split == "train" else a.stride_val
        videos = [v for v in (load_video(mot_root, s) for s in split_cfg[split]) if v is not None]
        missing = sorted(set(split_cfg[split]) - {v.seq for v in videos})
        if missing:
            raise RuntimeError(f"{split}: videos not found under {mot_root}: {missing}")
        jobs = [{"video": v, "tile": a.tile, "overlap": a.overlap, "stride": stride,
                 "min_inside": a.min_inside} for v in videos]
        with ProcessPoolExecutor(a.workers) as ex:
            scanned = [r for res in ex.map(scan_video, jobs) for r in res]
        n_frames = sum(len(sampled_frames(len(v.frames), stride)) for v in videos)

        usable = [r for r in scanned if r["black"] < a.max_black]
        n_black = {"positive": sum(1 for r in scanned if r["black"] >= a.max_black and len(r["cls"])),
                   "negative": sum(1 for r in scanned if r["black"] >= a.max_black and not len(r["cls"]))}
        pos = [r for r in usable if len(r["cls"])]
        neg = [r for r in usable if not len(r["cls"])]
        n_neg = min(len(neg), int(round(a.neg_ratio * len(pos))))
        kept = pos + rng.sample(neg, n_neg)

        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        by_seq: dict[str, list[dict]] = defaultdict(list)
        for r in kept:
            by_seq[r["seq"]].append(r)
        wjobs = [{"video": v, "tiles": by_seq[v.seq], "img_dir": img_dir, "lbl_dir": lbl_dir,
                  "jpeg_quality": a.jpeg_quality} for v in videos if by_seq[v.seq]]
        with ProcessPoolExecutor(a.workers) as ex:
            written = sum(ex.map(write_video, wjobs))
        assert written == len(kept), (written, len(kept))

        st = split_stats(scanned, kept, n_frames, len(videos), n_black)
        st["negatives_available"] = len(neg)
        st["per_video"] = {
            v.seq: {"tiles": len(by_seq[v.seq]),
                    "plane": int(sum((r["cls"] == 0).sum() for r in by_seq[v.seq])),
                    "ship": int(sum((r["cls"] == 1).sum() for r in by_seq[v.seq]))}
            for v in videos
        }
        stats["splits"][split] = st
        print(f"[tiles] {split}: {len(videos)} videos, {n_frames} frames, {len(kept)} tiles "
              f"({len(pos)} pos / {n_neg} neg), boxes {st['boxes']}, black-dropped {n_black}", flush=True)

    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {k}: {v}\n" for k, v in CLASSES.items())
    )
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="dataset folder name under --out-root")
    ap.add_argument("--mot-root", default="~/talos/data/mot")
    ap.add_argument("--out-root", default="~/talos/data/yolo")
    ap.add_argument("--split-file", default="configs/splits/airmot.json")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--stride-train", type=int, default=5)
    ap.add_argument("--stride-val", type=int, default=10)
    ap.add_argument("--min-inside", type=float, default=0.5, help="min box area fraction inside a tile")
    ap.add_argument("--max-black", type=float, default=0.9, help="drop tiles at least this black")
    ap.add_argument("--neg-ratio", type=float, default=0.2, help="negatives per positive tile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true", help="delete an existing build of --name first")
    return ap.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
