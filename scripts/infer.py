"""Run EndoCaver inference on a folder of images and save masks/restored RGB."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from endocaver import EndoCaver


def load_image(path: Path, size: int = 224):
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return img, tensor


def save_tensor_rgb(tensor, path: Path):
    arr = tensor.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255).astype("uint8")).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Input image folder")
    parser.add_argument("--checkpoint", default="", help="Optional .pt checkpoint")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true", help="Load the MiT-B0 encoder from local Hugging Face cache only")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "restored").mkdir(parents=True, exist_ok=True)

    model = EndoCaver(local_files_only=args.local_files_only).to(args.device).eval()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=True)

    image_paths = sorted([p for p in Path(args.images).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    with torch.no_grad():
        for path in image_paths:
            _, x = load_image(path)
            x = x.to(args.device)
            restored, mask = model(x)
            save_tensor_rgb(restored, out / "restored" / f"{path.stem}.png")
            m = (mask.squeeze().detach().cpu().numpy() > 0.5).astype("uint8") * 255
            Image.fromarray(m).save(out / "masks" / f"{path.stem}.png")
            print(path.name)


if __name__ == "__main__":
    main()
