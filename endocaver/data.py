"""Small Kvasir-style dataset helpers for EndoCaver examples."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class KvasirEndoCaverDataset(Dataset):
    """Return degraded image, clean image, binary mask.

    Expected directory layout:
        Kvasir-SEG/
          images/*.jpg or *.png
          masks/*.jpg or *.png
    """

    def __init__(self, root: str, image_size: int = 224, training: bool = False):
        self.root = Path(root)
        image_dir = self.root / "images"
        mask_dir = self.root / "masks"
        self.images = sorted(glob.glob(str(image_dir / "*")))
        self.masks = sorted(glob.glob(str(mask_dir / "*")))
        if not self.images or len(self.images) != len(self.masks):
            raise RuntimeError(f"Could not pair images/masks under {self.root}")

        common = [A.Resize(image_size, image_size)]
        if training:
            common += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2), A.Rotate(limit=15, p=0.5)]
        self.common = A.Compose(common)
        self.degrade = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.2, 0.2), p=1.0),
            A.MotionBlur(blur_limit=(3, 29), p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.ImageCompression(quality_lower=35, quality_upper=80, p=0.5),
            A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.55, p=0.25),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]).convert("L"))
        aug = self.common(image=image, mask=mask)
        clean = aug["image"]
        mask = (aug["mask"] > 127).astype("float32")
        degraded = self.degrade(image=clean)["image"]

        degraded = torch.from_numpy(degraded).permute(2, 0, 1).float() / 255.0
        clean = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return degraded, clean, mask
