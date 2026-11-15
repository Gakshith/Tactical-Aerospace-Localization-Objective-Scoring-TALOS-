"""ONNX Runtime detector backend: the same call contract as ``tiled.UltralyticsBackend``
(list of HxWx3 uint8 BGR images -> list of (k, 6) [x1, y1, x2, y2, conf, cls] in each
image's own pixels), with no torch and no ultralytics at run time.

The graph must carry the YOLO26 end2end (one-to-one, NMS-free) head: one output of
shape (B, max_det, 6) = [x1, y1, x2, y2, score, class] in model-input pixels, already
top-k selected. A raw one-to-many output ((B, 4 + nc, anchors), needs NMS) is refused,
because running it without NMS would be wrong and with NMS would be a different model
(on ultralytics 8.4.210 that is the silent ~10x-fewer-boxes failure).

Two input layouts are accepted, read from the graph:
  * ``float32 (B, 3, S, S)``: the stock Ultralytics export. Pre-processing here is the
    Ultralytics predictor's: letterbox to S x S (aspect kept, centre pad 114, bilinear
    resize, ``auto=False`` as for any fixed-size export), BGR -> RGB, HWC -> CHW, /255.
  * ``uint8 (B, S, S, 3)`` BGR: a graph with that conversion fused in front
    (``export_onnx.py --fuse-preprocess``). The letterbox still runs here; only the
    cast / flip / transpose / divide move into the graph, so 4x fewer bytes go to the
    device and the CPU skips a float pass. Same arithmetic: Cast then Div by 255.
Post-processing is Ultralytics' end2end branch: keep rows with ``score > conf``
(strict), optional class filter, first ``max_det`` rows, then undo the letterbox and
clip to the image (``ops.scale_boxes``).

Providers: ``cpu``, ``cuda`` (fp32), ``tensorrt`` (fp16 with ``fp16=True``; engines
are cached under ``trt_cache_dir`` keyed by the graph, so only the first run builds).
A graph with a fixed batch is fed in chunks of that batch, the last chunk zero-padded.
"""
from __future__ import annotations

import ast
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PROVIDERS = ("cpu", "cuda", "tensorrt")


def letterbox_params(h: int, w: int, size: int) -> tuple[float, int, int, int, int, int, int]:
    """Ultralytics ``LetterBox(auto=False, center=True, scaleup=True)`` geometry for an
    h x w image into size x size: (r, new_w, new_h, top, bottom, left, right)."""
    r = min(size / h, size / w)
    nw, nh = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    return r, nw, nh, round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)


def letterbox(im: np.ndarray, size: int) -> np.ndarray:
    h, w = im.shape[:2]
    if (h, w) == (size, size):
        return im
    _, nw, nh, top, bottom, left, right = letterbox_params(h, w, size)
    if (w, h) != (nw, nh):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def unletterbox(boxes: np.ndarray, h: int, w: int, size: int) -> np.ndarray:
    """Model-input xyxy -> image xyxy, as ``ultralytics.utils.ops.scale_boxes`` (in place, clipped)."""
    if (h, w) != (size, size):
        gain = min(size / h, size / w)
        nh, nw = max(round(h * gain), 1), max(round(w * gain), 1)
        gy, gx = nh / h, nw / w
        px, py = round((size - nw) / 2 - 0.1), round((size - nh) / 2 - 0.1)
        boxes[:, [0, 2]] -= px
        boxes[:, [1, 3]] -= py
        boxes[:, [0, 2]] /= gx
        boxes[:, [1, 3]] /= gy
    boxes[:, 0] = boxes[:, 0].clip(0, w)
    boxes[:, 2] = boxes[:, 2].clip(0, w)
    boxes[:, 1] = boxes[:, 1].clip(0, h)
    boxes[:, 3] = boxes[:, 3].clip(0, h)
    return boxes


def end2end_filter(pred: np.ndarray, conf: float, classes, max_det: int) -> np.ndarray:
    """Ultralytics ``nms.non_max_suppression`` end2end branch on one image's (N, 6) rows."""
    m = pred[:, 4] > conf
    if classes is not None:
        m &= np.isin(pred[:, 5], np.asarray(classes, pred.dtype))
    return pred[m][:max_det].astype(np.float32)


@dataclass
class OnnxBackend:
    """``onnxruntime.InferenceSession`` over a YOLO26 end2end graph. See the module docstring."""
    model: str
    provider: str = "cpu"             # cpu | cuda | tensorrt
    imgsz: int | None = None          # None = from the graph (static) or its Ultralytics metadata
    conf: float = 0.1
    classes: list[int] | None = None
    max_det: int = 300
    device_id: int = 0
    threads: int | None = None        # CPU intra-op threads (None = ORT default: all cores)
    fp16: bool = False                # tensorrt only
    trt_cache_dir: str | None = None
    trt_max_batch: int = 64           # tensorrt optimisation profile for a dynamic batch dim
    session_options: dict = field(default_factory=dict)
    speed_ms: dict = field(default_factory=lambda: {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0})
    images_seen: int = 0

    def __post_init__(self):
        import onnxruntime as ort

        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}, got {self.provider!r}")
        if self.provider != "cpu" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # CUDA / cuDNN / TensorRT from the nvidia-* wheels, if installed
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.threads:
            so.intra_op_num_threads = int(self.threads)
            so.inter_op_num_threads = 1
        for k, v in self.session_options.items():
            setattr(so, k, v)
        self._ort_version = ort.__version__
        self.sess = ort.InferenceSession(str(self.model), so, providers=self._providers(ort))
        active = self.sess.get_providers()
        want = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider",
                "tensorrt": "TensorrtExecutionProvider"}[self.provider]
        if want not in active:
            raise RuntimeError(f"{want} did not load (active: {active}); check the CUDA/TensorRT libraries")
        self.active_providers = active

        inp = self.sess.get_inputs()
        outs = self.sess.get_outputs()
        if len(inp) != 1:
            raise ValueError(f"expected one input, got {[i.name for i in inp]}")
        self.in_name, shape, dtype = inp[0].name, list(inp[0].shape), inp[0].type
        self.uint8_nhwc = dtype == "tensor(uint8)"
        if self.uint8_nhwc:
            if shape[-1] != 3:
                raise ValueError(f"uint8 input must be (B, S, S, 3) BGR, got {shape}")
            spatial = shape[1:3]
        else:
            if dtype != "tensor(float)" or shape[1] != 3:
                raise ValueError(f"float input must be (B, 3, S, S), got {dtype} {shape}")
            spatial = shape[2:4]
        self.fixed_batch = shape[0] if isinstance(shape[0], int) else None
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.metadata = dict(meta)
        size = None
        if all(isinstance(s, int) for s in spatial):
            if spatial[0] != spatial[1]:
                raise ValueError(f"square inputs only, got {spatial}")
            size = spatial[0]
        elif "imgsz" in meta:
            size = int(ast.literal_eval(meta["imgsz"])[0])  # Ultralytics writes e.g. "[1024, 1024]"
        if self.imgsz is None:
            if size is None:
                raise ValueError("graph has dynamic H/W and no imgsz metadata; pass imgsz")
            self.imgsz = size
        elif size is not None and all(isinstance(s, int) for s in spatial) and size != self.imgsz:
            raise ValueError(f"imgsz={self.imgsz} but the graph is fixed at {size}")

        # end2end check: one output (B, K, 6). The one-to-many head is (B, 4 + nc, anchors).
        if len(outs) != 1 or len(outs[0].shape) != 3 or outs[0].shape[-1] != 6:
            raise ValueError(f"not an end2end (NMS-free) graph: outputs {[(o.name, o.shape) for o in outs]}; "
                             "export with nms=False (end2end head)")
        self.out_name = outs[0].name
        self.k = outs[0].shape[1] if isinstance(outs[0].shape[1], int) else None
        if self.metadata.get("end2end", "True") not in ("True", "true", "1"):
            raise ValueError(f"graph metadata says end2end={self.metadata.get('end2end')}")

    def _providers(self, ort):
        cpu = "CPUExecutionProvider"
        if self.provider == "cpu":
            return [cpu]
        cuda = ("CUDAExecutionProvider", {"device_id": self.device_id})
        if self.provider == "cuda":
            return [cuda, cpu]
        cache = self.trt_cache_dir or str(Path(self.model).with_suffix("")) + "_trt_cache"
        os.makedirs(cache, exist_ok=True)
        opts = {"device_id": self.device_id, "trt_fp16_enable": bool(self.fp16),
                "trt_engine_cache_enable": True, "trt_engine_cache_path": cache,
                "trt_timing_cache_enable": True, "trt_timing_cache_path": cache,
                "trt_builder_optimization_level": 3}
        prof = self._trt_profile()
        if prof:
            opts.update(prof)
        return [("TensorrtExecutionProvider", opts), cuda, cpu]

    def _trt_profile(self) -> dict:
        """Explicit shape profile for a dynamic-batch graph (else TRT rebuilds per new shape)."""
        import onnx

        g = onnx.load(str(self.model), load_external_data=False).graph
        i = g.input[0]
        dims = [d.dim_value if d.HasField("dim_value") else None for d in i.type.tensor_type.shape.dim]
        if dims[0] is not None:
            return {}
        u8 = i.type.tensor_type.elem_type == onnx.TensorProto.UINT8
        s = self.imgsz or (dims[1] if u8 else dims[2])
        if s is None:
            meta = {p.key: p.value for p in onnx.load(str(self.model), load_external_data=False).metadata_props}
            s = int(ast.literal_eval(meta["imgsz"])[0])
        tail = f"{s}x{s}x3" if u8 else f"3x{s}x{s}"
        n = i.name
        return {"trt_profile_min_shapes": f"{n}:1x{tail}",
                "trt_profile_opt_shapes": f"{n}:{self.trt_max_batch}x{tail}",
                "trt_profile_max_shapes": f"{n}:{self.trt_max_batch}x{tail}"}

    def _blob(self, ims: list[np.ndarray]) -> np.ndarray:
        x = np.stack(ims)  # (B, S, S, 3) uint8 BGR
        if self.uint8_nhwc:
            return x
        x = x[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32)  # RGB CHW, contiguous copy
        np.divide(x, 255.0, out=x)
        return x

    def __call__(self, images) -> list[np.ndarray]:
        if len(images) == 0:
            return []
        S = self.imgsz
        out: list[np.ndarray] = []
        step = self.fixed_batch or len(images)
        if self.provider == "tensorrt" and self.fixed_batch is None:
            step = min(step, self.trt_max_batch)
        for s in range(0, len(images), step):
            chunk = images[s:s + step]
            t0 = time.perf_counter()
            ims = [letterbox(np.ascontiguousarray(im), S) for im in chunk]
            if self.fixed_batch and len(ims) < self.fixed_batch:
                pad = np.zeros_like(ims[0])
                ims = ims + [pad] * (self.fixed_batch - len(ims))
            blob = self._blob(ims)
            t1 = time.perf_counter()
            pred = self.sess.run([self.out_name], {self.in_name: blob})[0]
            t2 = time.perf_counter()
            for j, im in enumerate(chunk):
                d = end2end_filter(np.asarray(pred[j], np.float32), self.conf, self.classes, self.max_det)
                if len(d):
                    d[:, :4] = unletterbox(d[:, :4].copy(), im.shape[0], im.shape[1], S)
                out.append(d)
            t3 = time.perf_counter()
            self.speed_ms["preprocess"] += 1000 * (t1 - t0)
            self.speed_ms["inference"] += 1000 * (t2 - t1)
            self.speed_ms["postprocess"] += 1000 * (t3 - t2)
        self.images_seen += len(images)
        return out

    def describe(self) -> dict:
        return {"runtime": "onnxruntime", "onnxruntime": self._ort_version, "model": Path(self.model).name,
                "provider": self.provider, "active_providers": self.active_providers,
                "fp16": self.fp16 if self.provider == "tensorrt" else False,
                "threads": self.threads, "input": self.in_name,
                "input_layout": "uint8 BHWC BGR (fused preprocess)" if self.uint8_nhwc else "float32 BCHW RGB /255",
                "fixed_batch": self.fixed_batch, "imgsz_arg": self.imgsz, "imgsz_used": [self.imgsz, self.imgsz],
                "end2end_arg": True, "head_end2end_active": True, "output_topk": self.k,
                "predict_kwargs": {"conf": self.conf, "max_det": self.max_det,
                                   **({"classes": list(self.classes)} if self.classes is not None else {})}}
