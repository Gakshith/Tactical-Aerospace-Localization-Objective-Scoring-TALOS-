"""Sliced ("tiled", SAHI-style) inference for tiny objects in full satellite frames.

A 1920x1080 frame is cut into overlapping native-resolution tiles (default
640 px, overlap 0.2), every tile goes through the detector at its own
resolution, and the per-tile boxes are shifted back to frame pixels and merged.
Optionally one extra pass runs on the whole frame downscaled to ``full_imgsz``
(SAHI's "standard prediction") so objects larger than the tile overlap, which
every tile sees cut, are still found whole.

Tile grid
    Per axis: starts 0, s, 2s, ... with stride s = round(tile * (1 - overlap)),
    and the last tile flush with the far edge (start = L - tile). A frame
    shorter than ``tile`` gets one tile of the frame's length. 1920x1080 at 640 /
    0.2 -> x 0,512,1024,1280 and y 0,440: 8 tiles per frame.

Merge rule (class-aware, per frame)
    Each box carries its source (tile index, or the full-frame pass) and four
    "cut" flags: it touches an *interior* tile edge (within ``edge_margin`` px;
    frame borders never count) on that side, so it may be a fragment of an
    object the neighbouring tile sees whole.

    Greedy clustering, anchors taken uncut-first, then by confidence. An anchor
    absorbs same-class boxes from *other* sources that
      (a) duplicate it: IoU >= ``iou_thr``, or each box's centre lies inside the
          other (for 3-10 px ships a 1-px disagreement between two tiles drops
          IoU below 0.5; see Validation below), or
      (b) are cut and lie mostly inside it, IoS = inter / area(member) >=
          ``ios_thr``, or have their centre inside it (the fragment of an
          object another tile saw whole).
    If the anchor is itself cut (no source saw the object whole: it is larger
    than the overlap and there is no full-frame pass), cut boxes from other
    tiles that continue it across the edge it is cut on (they overlap it, are
    cut on the facing side, and their extents along that edge have 1-D IoU >=
    ``span_iou``) are merged into the union box, SAHI-NMM style; this repeats,
    so an object over a 2x2 tile corner becomes one box. The cluster's output is
    the anchor geometry (or the union) with the cluster's max confidence.

    Why this and not plain NMS or plain NMM: NMS with IoU alone keeps fragments
    (a 10-px sliver of a 30-px plane has IoU ~0.3 with the whole box), and
    SAHI's NMM unions every IoS match, which inflates boxes when two distinct
    small ships overlap a little. Here geometry is only ever grown from pieces
    that are provably cut by the tiling, and only the fragment (never a whole,
    uncut box) is dropped by containment, so moored ships packed side by side
    are left alone. Boxes from the same source are never merged with each
    other: the model's own output for one image is already deduplicated (the
    YOLO26 end2end one-to-one head, or NMS), and re-suppressing it would delete
    real neighbours in dense harbours.

    Validation on real geometry (all 3,148 AIR-MOT val frames, both benches,
    25,377 GT boxes, 4,346 of them crossing an interior tile edge), with an
    oracle backend that reports each GT box clipped to the tile: with exact
    boxes the merge returns exactly one box per GT, every one matched (IoU >=
    0.5), zero extras. With 1-px Gaussian box noise per tile it returns 0.6 %
    more boxes than the same noise on whole, untiled frames (IoU-only dedup
    gave 2.1 %), and recall is not lower than untiled.

Implemented directly, not with the ``sahi`` library (0.14.0 on Polaris):
    sahi's ``get_sliced_prediction`` takes one image; it batches the slices of
    that image (``batch_size``) but not across frames, and its Ultralytics
    wrapper calls the model with fixed kwargs (verbose, conf, device, imgsz), so
    on ultralytics 8.4.210 a YOLO26 checkpoint silently runs the one-to-many
    head + NMS unless sahi's internals are patched. Its merge (GREEDYNMM / NMS
    on IoS or IoU) also has no notion of which boxes a slice edge cut. Here the
    tiles of several frames go through ``YOLO.predict`` as one list (64 tiles
    per call by default), with ``end2end=True`` passed exactly as the full-frame
    runner does.

    The detector is pluggable (``backend`` / ``full_backend``: any callable
    images -> list of (k, 6) arrays), so tests run with a fake model, no GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# cut-flag columns
L, T, R, B = 0, 1, 2, 3
FULL_SRC = -1  # source id of the downscaled full-frame pass


def axis_starts(length: int, tile: int, overlap: float) -> tuple[list[int], int]:
    """Tile start offsets along one axis and the tile length used on it."""
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if length <= tile:
        return [0], length
    step = max(1, int(round(tile * (1 - overlap))))
    starts = list(range(0, length - tile, step)) + [length - tile]
    return starts, tile


def tile_grid(width: int, height: int, tile: int, overlap: float) -> np.ndarray:
    """(n, 4) int array of tiles [x0, y0, x1, y1] covering a width x height frame, row-major."""
    xs, tw = axis_starts(width, tile, overlap)
    ys, th = axis_starts(height, tile, overlap)
    return np.array([[x, y, x + tw, y + th] for y in ys for x in xs], dtype=np.int64)


def cut_flags(boxes: np.ndarray, tile: np.ndarray, width: int, height: int, margin: float) -> np.ndarray:
    """(k, 4) bool [L, T, R, B]: box touches that tile edge and the edge is interior to the frame."""
    x0, y0, x1, y1 = tile
    f = np.zeros((len(boxes), 4), bool)
    if len(boxes) == 0:
        return f
    f[:, L] = (x0 > 0) & (boxes[:, 0] <= x0 + margin)
    f[:, T] = (y0 > 0) & (boxes[:, 1] <= y0 + margin)
    f[:, R] = (x1 < width) & (boxes[:, 2] >= x1 - margin)
    f[:, B] = (y1 < height) & (boxes[:, 3] >= y1 - margin)
    return f


def _iou_ios(box: np.ndarray, others: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IoU of ``box`` with each of ``others`` and IoS = inter / area(other)."""
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    b = np.clip(others[:, 2] - others[:, 0], 0, None) * np.clip(others[:, 3] - others[:, 1], 0, None)
    union = a + b - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    ios = np.where(b > 0, inter / np.maximum(b, 1e-12), 0.0)
    return iou, ios


def _centre_inside(pts_boxes: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Centre of each of ``pts_boxes`` inside the matching (or broadcast) box of ``boxes``."""
    cx = (pts_boxes[..., 0] + pts_boxes[..., 2]) / 2
    cy = (pts_boxes[..., 1] + pts_boxes[..., 3]) / 2
    return (cx >= boxes[..., 0]) & (cx <= boxes[..., 2]) & (cy >= boxes[..., 1]) & (cy <= boxes[..., 3])


def _span_iou(a0, a1, b0, b1) -> np.ndarray:
    inter = np.clip(np.minimum(a1, b1) - np.maximum(a0, b0), 0, None)
    union = np.maximum(a1, b1) - np.minimum(a0, b0)
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def merge_tiles(boxes: np.ndarray, conf: np.ndarray, cls: np.ndarray, src: np.ndarray, cut: np.ndarray,
                iou_thr: float = 0.5, ios_thr: float = 0.6, span_iou: float = 0.5) -> np.ndarray:
    """Merge boxes from several tiles of one frame. Returns (k, 6) [x1, y1, x2, y2, conf, cls],
    sorted by conf descending. See the module docstring for the rule."""
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 6), np.float32)
    boxes = np.asarray(boxes, np.float64)
    conf = np.asarray(conf, np.float64)
    cls = np.asarray(cls)
    src = np.asarray(src)
    cut = np.asarray(cut, bool)
    alive = np.ones(n, bool)
    order = np.lexsort((-conf, cut.any(1)))  # uncut first, then conf desc
    out = []
    for i in order:
        if not alive[i]:
            continue
        alive[i] = False
        box, flags, c = boxes[i].copy(), cut[i].copy(), conf[i]
        srcs = {int(src[i])}
        grew = True
        while grew:
            grew = False
            cand = np.flatnonzero(alive & (cls == cls[i]) & ~np.isin(src, list(srcs)))
            if len(cand) == 0:
                break
            iou, ios = _iou_ios(box, boxes[cand])
            m_in_a = _centre_inside(boxes[cand], box)
            a_in_m = _centre_inside(box[None], boxes[cand])
            # duplicate: IoU, or mutual centre containment (IoU of a 4-px box is too noisy to rely on)
            absorb = (iou >= iou_thr) | (m_in_a & a_in_m)
            # fragment: a cut box mostly inside the anchor, or centred in it
            absorb |= cut[cand].any(1) & ((ios >= ios_thr) | m_in_a)
            # continuation of a cut anchor across the edge it is cut on
            cont = np.zeros(len(cand), bool)
            if flags.any():
                m, mc = boxes[cand], cut[cand]
                ox = (m[:, 0] < box[2]) & (box[0] < m[:, 2])
                oy = (m[:, 1] < box[3]) & (box[1] < m[:, 3])
                sy = _span_iou(box[1], box[3], m[:, 1], m[:, 3]) >= span_iou
                sx = _span_iou(box[0], box[2], m[:, 0], m[:, 2]) >= span_iou
                cont = ox & oy & (
                    (flags[R] & mc[:, L] & (m[:, 2] > box[2]) & sy)
                    | (flags[L] & mc[:, R] & (m[:, 0] < box[0]) & sy)
                    | (flags[B] & mc[:, T] & (m[:, 3] > box[3]) & sx)
                    | (flags[T] & mc[:, B] & (m[:, 1] < box[1]) & sx))
            take = cand[absorb | cont]
            if len(take) == 0:
                break
            alive[take] = False
            srcs.update(int(s) for s in src[take])
            c = max(c, float(conf[take].max()))
            for j in cand[cont]:
                box, flags = _union(box, flags, boxes[j], cut[j])
                grew = True
        out.append([*box, c, cls[i]])
    res = np.array(out, np.float32).reshape(-1, 6)
    return res[np.argsort(-res[:, 4], kind="stable")]


def _union(a, fa, b, fb):
    """Union box; each side keeps the cut flag of the box that defines it (OR on a tie)."""
    u = np.array([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])])
    f = np.zeros(4, bool)
    for side, k in ((L, 0), (T, 1)):
        f[side] = (fa[side] and a[k] <= b[k]) or (fb[side] and b[k] <= a[k])
    for side, k in ((R, 2), (B, 3)):
        f[side] = (fa[side] and a[k] >= b[k]) or (fb[side] and b[k] >= a[k])
    return u, f


@dataclass
class UltralyticsBackend:
    """``YOLO.predict`` over a list of BGR images -> list of (k, 6) [x1, y1, x2, y2, conf, cls]
    in each image's own pixels. ``end2end=True`` selects the YOLO26 one-to-one head (on
    ultralytics 8.4.210 the default silently runs the one-to-many head + NMS)."""
    weights: str
    imgsz: int | None = None          # None = checkpoint default
    conf: float = 0.1
    end2end: bool | None = True       # None = library default
    device: str = "cuda:0"
    half: bool = False
    classes: list[int] | None = None  # model class ids to keep (filtered before max_det)
    max_det: int = 300
    speed_ms: dict = field(default_factory=lambda: {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0})
    images_seen: int = 0

    def __post_init__(self):
        from ultralytics import YOLO
        self.model = YOLO(str(self.weights))
        kw = dict(conf=self.conf, device=self.device, verbose=False, max_det=self.max_det)
        if self.half:  # only when set: 8.4.210 logs a deprecation warning on every call that passes it
            kw["half"] = True
        if self.imgsz:
            kw["imgsz"] = self.imgsz
        if self.end2end is not None:
            kw["end2end"] = self.end2end
        if self.classes is not None:
            kw["classes"] = list(self.classes)
        self.kw = kw

    def __call__(self, images) -> list[np.ndarray]:
        """``images``: list of HxWx3 uint8 BGR arrays, or a (B, 3, H, W) float RGB tensor in [0, 1]
        already at the model's input size (no letterbox is applied to a tensor)."""
        if len(images) == 0:
            return []
        res = self.model.predict(images, **self.kw)
        out = []
        for r in res:
            for k in self.speed_ms:
                self.speed_ms[k] += r.speed[k]
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 6), np.float32))
                continue
            out.append(np.column_stack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                        b.cls.cpu().numpy()]).astype(np.float32))
        self.images_seen += len(images)
        return out

    def describe(self) -> dict:
        p = self.model.predictor
        imgsz = getattr(p, "imgsz", None)
        if isinstance(imgsz, (list, tuple)):
            imgsz = [int(v) for v in imgsz]
        return {"imgsz_arg": self.imgsz, "imgsz_used": imgsz,
                "end2end_arg": self.end2end,
                # the head the predictor actually ran (AutoBackend copy), not the checkpoint attr
                "head_end2end_active": bool(getattr(p.model, "end2end", False)) if p is not None else None,
                "predict_kwargs": {k: v for k, v in self.kw.items() if k != "device"}}


class TiledDetector:
    """Sliced inference over full frames. ``predict(frames)`` -> list of (k, 6)
    [x1, y1, x2, y2, conf, cls] in full-frame pixels, one array per frame.

    ``batch`` is the number of images (tiles) per detector call; tiles of several
    frames share a call. Pass ``backend`` / ``full_backend`` to inject a model
    (tests); otherwise Ultralytics backends are built from ``weights``.
    """

    def __init__(self, weights=None, tile: int = 640, overlap: float = 0.2, imgsz: int | None = None,
                 conf: float = 0.1, end2end: bool | None = True, device: str = "cuda:0", batch: int = 64,
                 with_full_frame: bool = False, full_imgsz: int | None = None, half: bool = False,
                 classes: list[int] | None = None, max_det: int = 300,
                 iou_thr: float = 0.5, ios_thr: float = 0.6, span_iou: float = 0.5, edge_margin: float = 3.0,
                 gpu_tiling: bool = False, backend=None, full_backend=None):
        self.tile, self.overlap, self.batch = int(tile), float(overlap), int(batch)
        self.device = device
        self.with_full_frame = with_full_frame
        self.iou_thr, self.ios_thr, self.span_iou, self.edge_margin = iou_thr, ios_thr, span_iou, edge_margin
        self.imgsz = imgsz or self.tile
        self.full_imgsz = full_imgsz or self.imgsz
        common = dict(conf=conf, end2end=end2end, device=device, half=half, classes=classes, max_det=max_det)
        self.backend = backend or UltralyticsBackend(weights, imgsz=self.imgsz, **common)
        self.full_backend = None
        if with_full_frame:
            # separate model instance: one predictor per input size, no re-setup between calls
            self.full_backend = full_backend or UltralyticsBackend(weights, imgsz=self.full_imgsz, **common)
        self.timing = {"tile_infer_s": 0.0, "full_infer_s": 0.0, "merge_s": 0.0, "frames": 0, "tiles": 0}
        # Optional: crop tiles on the GPU and hand the model a ready (B,3,T,T) tensor. Off by default:
        # on the A100 it removed ~5 ms/frame of preprocess but ultralytics then copies the tensor batch
        # back to numpy in postprocess (+~5 ms/frame), so no net gain (2026-11-11). Same boxes as the
        # list path on CPU. Only valid at native size (imgsz == tile, a multiple of 32): tensors are not resized.
        self.gpu_tiling = (gpu_tiling and isinstance(self.backend, UltralyticsBackend)
                           and self.imgsz == self.tile and self.tile % 32 == 0 and str(device).startswith("cuda"))

    def grid(self, width: int, height: int) -> np.ndarray:
        return tile_grid(width, height, self.tile, self.overlap)

    def predict(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        jobs = []  # (frame idx, tile idx)
        crops = []
        grids = []
        for fi, fr in enumerate(frames):
            h, w = fr.shape[:2]
            g = self.grid(w, h)
            grids.append(g)
            for ti, (x0, y0, x1, y1) in enumerate(g):
                jobs.append((fi, ti))
                crops.append((fi, x0, y0, x1, y1))
        t0 = time.perf_counter()
        raw = []
        full_tiles = all(x1 - x0 == self.tile and y1 - y0 == self.tile for _, x0, y0, x1, y1 in crops)
        if self.gpu_tiling and full_tiles and crops:
            import torch
            gpu = [torch.from_numpy(np.ascontiguousarray(fr)).to(self.device, non_blocking=True) for fr in frames]
            for s in range(0, len(crops), self.batch):
                t = torch.stack([gpu[fi][y0:y1, x0:x1] for fi, x0, y0, x1, y1 in crops[s:s + self.batch]])
                t = t.flip(-1).permute(0, 3, 1, 2).float().div_(255.0)  # BGR HWC uint8 -> RGB CHW [0,1]
                raw.extend(self.backend(t))
        else:
            ims = [frames[fi][y0:y1, x0:x1] for fi, x0, y0, x1, y1 in crops]
            for s in range(0, len(ims), self.batch):
                raw.extend(self.backend(ims[s:s + self.batch]))
        t1 = time.perf_counter()
        full = [None] * len(frames)
        if self.full_backend is not None:
            full = []
            fb = max(1, self.batch // max(len(grids[0]), 1))
            for s in range(0, len(frames), fb):
                full.extend(self.full_backend(list(frames[s:s + fb])))
        t2 = time.perf_counter()

        per_frame = [[] for _ in frames]
        for (fi, ti), d in zip(jobs, raw):
            if len(d):
                per_frame[fi].append((ti, d))
        out = []
        for fi, fr in enumerate(frames):
            h, w = fr.shape[:2]
            bx, cf, cl, sr, ct = [], [], [], [], []
            for ti, d in per_frame[fi]:
                t = grids[fi][ti]
                b = d[:, :4].astype(np.float64) + np.array([t[0], t[1], t[0], t[1]], np.float64)
                bx.append(b); cf.append(d[:, 4]); cl.append(d[:, 5]); sr.append(np.full(len(d), ti))
                ct.append(cut_flags(b, t, w, h, self.edge_margin))
            if full[fi] is not None and len(full[fi]):
                d = full[fi]
                bx.append(d[:, :4].astype(np.float64)); cf.append(d[:, 4]); cl.append(d[:, 5])
                sr.append(np.full(len(d), FULL_SRC)); ct.append(np.zeros((len(d), 4), bool))
            if not bx:
                out.append(np.zeros((0, 6), np.float32))
                continue
            out.append(merge_tiles(np.concatenate(bx), np.concatenate(cf), np.concatenate(cl),
                                   np.concatenate(sr), np.concatenate(ct),
                                   self.iou_thr, self.ios_thr, self.span_iou))
        t3 = time.perf_counter()
        self.timing["tile_infer_s"] += t1 - t0
        self.timing["full_infer_s"] += t2 - t1
        self.timing["merge_s"] += t3 - t2
        self.timing["frames"] += len(frames)
        self.timing["tiles"] += len(crops)
        return out

    def describe(self) -> dict:
        d = {"tile": self.tile, "overlap": self.overlap, "imgsz": self.imgsz, "batch_tiles": self.batch,
             "gpu_tiling": self.gpu_tiling,
             "with_full_frame": self.with_full_frame, "full_imgsz": self.full_imgsz if self.with_full_frame else None,
             "merge": {"rule": "class-aware greedy clusters, uncut-first; dup = IoU or mutual centre containment; "
                               "cut fragment = IoS or centre inside; union of cut continuations; cross-source only",
                       "iou_thr": self.iou_thr, "ios_thr": self.ios_thr, "span_iou": self.span_iou,
                       "edge_margin_px": self.edge_margin}}
        if hasattr(self.backend, "describe"):
            d["tile_backend"] = self.backend.describe()
        if self.full_backend is not None and hasattr(self.full_backend, "describe"):
            d["full_backend"] = self.full_backend.describe()
        return d
