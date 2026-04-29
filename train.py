"""Minimal EndoCaver training example on Kvasir-SEG.

This is intentionally compact: it is a clean single-file example for checking
that the model and data format work, not the full private training framework.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from endocaver import EndoCaver
from endocaver.data import KvasirEndoCaverDataset


def dice_loss(pred, target, eps=1e-6):
    pred = pred.flatten(1)
    target = target.flatten(1)
    inter = (pred * target).sum(1)
    denom = pred.sum(1) + target.sum(1)
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def locos_weight(epoch: int, total_epochs: int):
    """Small public version of LoCoS: shift focus from restoration to segmentation."""
    t = epoch / max(1, total_epochs - 1)
    deblur_w = 0.5 * (1.0 + math.cos(math.pi * t))
    seg_w = 1.0 - 0.5 * deblur_w
    return deblur_w, seg_w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Kvasir-SEG root containing images/ and masks/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="checkpoints/EndoCaver_MiTB0_GAM_DSA_LoCoS_Kvasir.pt")
    args = parser.parse_args()

    dataset = KvasirEndoCaverDataset(args.data, image_size=224, training=True)
    val_len = max(1, int(0.2 * len(dataset)))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = EndoCaver().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_dice = 0.0

    for epoch in range(args.epochs):
        model.train()
        deblur_w, seg_w = locos_weight(epoch, args.epochs)
        running = 0.0
        for degraded, clean, mask in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            degraded, clean, mask = degraded.to(args.device), clean.to(args.device), mask.to(args.device)
            restored, pred = model(degraded)
            loss_deblur = F.l1_loss(restored, clean)
            loss_seg = F.binary_cross_entropy(pred, mask) + dice_loss(pred, mask)
            loss = deblur_w * loss_deblur + seg_w * loss_seg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()

        model.eval()
        dices = []
        with torch.no_grad():
            for degraded, _, mask in val_loader:
                degraded, mask = degraded.to(args.device), mask.to(args.device)
                _, pred = model(degraded)
                bin_pred = (pred > 0.5).float()
                inter = (bin_pred * mask).sum(dim=(1, 2, 3))
                denom = bin_pred.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
                dices.append(((2 * inter + 1e-6) / (denom + 1e-6)).mean().item())
        mean_dice = sum(dices) / max(1, len(dices))
        print(f"epoch={epoch+1} loss={running/max(1,len(train_loader)):.4f} val_dice={mean_dice:.4f}")
        if mean_dice > best_dice:
            best_dice = mean_dice
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "val_dice": best_dice}, args.out)
            print(f"saved {args.out}")


if __name__ == "__main__":
    main()
