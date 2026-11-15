"""Export a YOLO26 checkpoint for ONNX Runtime / TensorRT, keeping the end2end head.

    python -m src.TALOS.detect.export_onnx --weights ~/talos/weights/export/y26s_t1024.pt \
        --imgsz 1024 --onnx --fuse-preprocess --engine-half --engine-batch 48

Steps (each optional):
  --onnx             Ultralytics ONNX export, fp32, ``nms=False`` (= the one-to-one end2end
                     head; on ultralytics 8.4.210 ``nms=None`` would export the one-to-many
                     head and need NMS), dynamic batch/H/W, opset ``--opset`` (default 18,
                     Ultralytics' pick for torch>=2.4: ORT's CUDA EP lacks Resize-19 kernels).
                     -> <stem>.onnx
  --fuse-preprocess  Copy of that graph with the predictor's tensor pre-processing in front:
                     input uint8 (B, S, S, 3) BGR -> Cast float -> Transpose NCHW -> Gather
                     channels [2,1,0] (RGB) -> Div 255 -> the original input. Same arithmetic
                     as ultralytics' predictor (permute, flip, float, div). H/W fixed at S.
                     -> <stem>_u8.onnx
  --int8-dynamic     ORT dynamic INT8 quantisation (weights int8, activations quantised at run
                     time) of <stem>.onnx -> <stem>_int8dyn.onnx. CPU only.
  --engine-half      Ultralytics TensorRT export, fp16, dynamic batch up to --engine-batch.
                     -> <stem>.engine (needs the ``tensorrt`` package and a GPU)
Every artifact's I/O shapes, opset, metadata and sha256 go to <dir>/export_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_onnx(p: Path) -> dict:
    import onnx

    m = onnx.load(str(p), load_external_data=False)

    def shp(v):
        t = v.type.tensor_type
        return {"name": v.name, "dtype": onnx.TensorProto.DataType.Name(t.elem_type),
                "shape": [d.dim_value if d.HasField("dim_value") else d.dim_param for d in t.shape.dim]}

    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return {"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p),
            "opset": {o.domain or "ai.onnx": o.version for o in m.opset_import},
            "ir_version": m.ir_version, "producer": f"{m.producer_name} {m.producer_version}",
            "inputs": [shp(v) for v in m.graph.input], "outputs": [shp(v) for v in m.graph.output],
            "metadata": {q.key: q.value for q in m.metadata_props},
            "nodes": len(m.graph.node), "has_TopK": "TopK" in ops,
            "quant_ops": {k: v for k, v in ops.items() if "Integer" in k or "Quant" in k}}


def fuse_preprocess(src: Path, dst: Path, size: int) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    m = onnx.load(str(src))
    g = m.graph
    old = g.input[0]
    name = old.name
    batch = old.type.tensor_type.shape.dim[0]
    bdim = batch.dim_param if not batch.HasField("dim_value") else batch.dim_value
    new_in = helper.make_tensor_value_info(f"{name}_u8", TensorProto.UINT8, [bdim, size, size, 3])
    idx = numpy_helper.from_array(np.array([2, 1, 0], np.int64), f"{name}_bgr2rgb_idx")
    scale = numpy_helper.from_array(np.array(255.0, np.float32), f"{name}_div255")
    nodes = [
        helper.make_node("Cast", [new_in.name], [f"{name}_f32"], to=TensorProto.FLOAT, name="pre_cast"),
        helper.make_node("Transpose", [f"{name}_f32"], [f"{name}_nchw"], perm=[0, 3, 1, 2], name="pre_nchw"),
        helper.make_node("Gather", [f"{name}_nchw", idx.name], [f"{name}_rgb"], axis=1, name="pre_rgb"),
        helper.make_node("Div", [f"{name}_rgb", scale.name], [name], name="pre_div255"),
    ]
    g.initializer.extend([idx, scale])
    for n in reversed(nodes):
        g.node.insert(0, n)
    g.input.remove(old)
    g.input.insert(0, new_in)
    meta = {p.key: p for p in m.metadata_props}
    if "imgsz" in meta:
        meta["imgsz"].value = str([size, size])
    p = m.metadata_props.add()
    p.key, p.value = "talos_preprocess", "fused: uint8 BHWC BGR -> float BCHW RGB /255 (letterbox outside)"
    m = onnx.shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True, help=".pt; artifacts are written next to it")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--fuse-preprocess", action="store_true")
    ap.add_argument("--int8-dynamic", action="store_true")
    ap.add_argument("--engine-half", action="store_true")
    ap.add_argument("--engine-batch", type=int, default=48)
    a = ap.parse_args()

    w = a.weights.resolve()
    stem = w.with_suffix("")
    man_p = w.parent / "export_manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    man["source_pt"] = {"file": w.name, "sha256": sha256(w)}
    onnx_p = stem.with_suffix(".onnx")

    if a.onnx:
        from ultralytics import YOLO
        import ultralytics, torch

        t = time.time()
        out = YOLO(str(w)).export(format="onnx", imgsz=a.imgsz, opset=a.opset, nms=False, dynamic=True,
                                  simplify=True, device="cpu")
        assert Path(out).resolve() == onnx_p, (out, onnx_p)
        man["onnx"] = {**describe_onnx(onnx_p), "export_s": round(time.time() - t, 1),
                       "export_args": {"format": "onnx", "imgsz": a.imgsz, "opset": a.opset, "nms": False,
                                       "dynamic": True, "simplify": True},
                       "ultralytics": ultralytics.__version__, "torch": torch.__version__}
    if a.fuse_preprocess:
        u8 = stem.parent / f"{stem.name}_u8.onnx"
        fuse_preprocess(onnx_p, u8, a.imgsz)
        man["onnx_u8"] = describe_onnx(u8)
    if a.int8_dynamic:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.shape_inference import quant_pre_process

        pre = stem.parent / f"{stem.name}_qpre.onnx"
        q = stem.parent / f"{stem.name}_int8dyn.onnx"
        quant_pre_process(str(onnx_p), str(pre), skip_symbolic_shape=True)
        quantize_dynamic(str(pre), str(q), weight_type=QuantType.QInt8)
        pre.unlink()
        man["onnx_int8dyn"] = describe_onnx(q)
    if a.engine_half:
        from ultralytics import YOLO
        import tensorrt

        # the engine export writes its own intermediate <stem>.onnx; keep ours intact
        bak = stem.parent / f"{stem.name}.onnx.keep"
        if onnx_p.exists():
            onnx_p.rename(bak)
        t = time.time()
        try:
            out = YOLO(str(w)).export(format="engine", imgsz=a.imgsz, quantize=16, nms=False, dynamic=True,
                                      batch=a.engine_batch, device=0, simplify=True)
        finally:
            if onnx_p.exists():
                onnx_p.rename(stem.parent / f"{stem.name}_engine_src.onnx")
            if bak.exists():
                bak.rename(onnx_p)
        e = Path(out).resolve()
        src = stem.parent / f"{stem.name}_engine_src.onnx"
        man["engine_fp16"] = {"file": e.name, "bytes": e.stat().st_size, "sha256": sha256(e),
                              "export_s": round(time.time() - t, 1), "tensorrt": tensorrt.__version__,
                              "export_args": {"format": "engine", "imgsz": a.imgsz, "quantize": 16, "nms": False,
                                              "dynamic": True, "batch": a.engine_batch},
                              "intermediate_onnx": describe_onnx(src) if src.exists() else None}
    man_p.write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
