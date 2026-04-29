"""Export EndoCaver checkpoint to ONNX with paper-aligned artifact names."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from endocaver import EndoCaver


class EndoCaverONNXWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image):
        deblur_output, segmentation_output = self.model(image)
        return deblur_output, segmentation_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/EndoCaver_MiTB0_GAM_DSA_LoCoS_Kvasir.pt")
    parser.add_argument("--out", default="checkpoints/EndoCaver_MiTB0_GAM_DSA_LoCoS_Kvasir.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--local-files-only", action="store_true", help="Load the MiT-B0 encoder from local Hugging Face cache only")
    args = parser.parse_args()

    torch.backends.mha.set_fastpath_enabled(False)
    model = EndoCaver(local_files_only=args.local_files_only).eval().cpu()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=True)
    wrapper = EndoCaverONNXWrapper(model).eval()
    example = torch.randn(1, 3, 224, 224)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        example,
        args.out,
        input_names=["image"],
        output_names=["deblur_output", "segmentation_output"],
        dynamic_axes={"image": {0: "batch"}, "deblur_output": {0: "batch"}, "segmentation_output": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    model_onnx = onnx.load(args.out)
    onnx.checker.check_model(model_onnx)
    print(f"ONNX exported and checked: {args.out}")


if __name__ == "__main__":
    main()
