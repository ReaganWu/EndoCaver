"""Compare EndoCaver PyTorch and ONNX Runtime outputs/latency.

Modes:
  pt          PyTorch checkpoint evaluation
  onnx-cpu    ONNX Runtime CPUExecutionProvider
  onnx-gpu    ONNX Runtime CUDAExecutionProvider if available
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from endocaver import EndoCaver


def load_image(path: str):
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype("float32") / 255.0
    return arr.transpose(2, 0, 1)[None, ...]


def summarize_outputs(deblur, seg):
    return {
        "deblur_shape": list(deblur.shape),
        "segmentation_shape": list(seg.shape),
        "deblur_min": float(deblur.min()),
        "deblur_max": float(deblur.max()),
        "segmentation_min": float(seg.min()),
        "segmentation_max": float(seg.max()),
        "segmentation_mean": float(seg.mean()),
    }


def eval_pt(args, image_np):
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = EndoCaver(local_files_only=args.local_files_only).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=True)
    x = torch.from_numpy(image_np).to(device)
    with torch.no_grad():
        for _ in range(args.warmup):
            out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.iters):
            out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    deblur, seg = [o.detach().cpu().numpy() for o in out]
    return "PyTorch", str(device), elapsed, deblur, seg


def eval_onnx(args, image_np, provider):
    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(f"Requested provider {provider} not available. Available providers: {available}")
    sess = ort.InferenceSession(args.onnx, providers=[provider])
    feed = {"image": image_np.astype("float32")}
    for _ in range(args.warmup):
        out = sess.run(None, feed)
    start = time.perf_counter()
    for _ in range(args.iters):
        out = sess.run(None, feed)
    elapsed = time.perf_counter() - start
    return "ONNX Runtime", provider, elapsed, out[0], out[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pt", "onnx-cpu", "onnx-gpu"], required=True)
    parser.add_argument("--image", default="samples/kvasir/images/kvasir_sample_01.png")
    parser.add_argument("--checkpoint", default="checkpoints/EndoCaver_MiTB0_GAM_DSA_LoCoS_Kvasir.pt")
    parser.add_argument("--onnx", default="checkpoints/EndoCaver_MiTB0_GAM_DSA_LoCoS_Kvasir.onnx")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out", default="docs/runtime_eval.json")
    parser.add_argument("--local-files-only", action="store_true", help="Load the MiT-B0 encoder from local Hugging Face cache only")
    args = parser.parse_args()

    image_np = load_image(args.image)
    if args.mode == "pt":
        runtime, provider, elapsed, deblur, seg = eval_pt(args, image_np)
    elif args.mode == "onnx-cpu":
        runtime, provider, elapsed, deblur, seg = eval_onnx(args, image_np, "CPUExecutionProvider")
    else:
        runtime, provider, elapsed, deblur, seg = eval_onnx(args, image_np, "CUDAExecutionProvider")

    result = {
        "mode": args.mode,
        "runtime": runtime,
        "provider_or_device": provider,
        "image": args.image,
        "iterations": args.iters,
        "avg_latency_ms": elapsed / args.iters * 1000.0,
        "fps": args.iters / elapsed,
        **summarize_outputs(deblur, seg),
    }
    print(json.dumps(result, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        if not isinstance(existing, list):
            existing = [existing]
    else:
        existing = []
    existing = [r for r in existing if r.get("mode") != args.mode] + [result]
    out_path.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
