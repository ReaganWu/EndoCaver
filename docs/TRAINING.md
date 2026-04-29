# Training

This public release keeps training intentionally simple. The included script is a compact Kvasir-SEG example for users who want to verify the architecture and data flow.

It is not a full reproduction of every private experiment script.

## Data

Use the standard Kvasir-SEG folder structure:

```text
Kvasir-SEG/
  images/
    image_001.jpg
    image_002.jpg
  masks/
    image_001.jpg
    image_002.jpg
```

Images are resized to 224 x 224. Masks are binarized internally.

## Run

```bash
python train.py \
  --data /path/to/Kvasir-SEG \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --out checkpoints/EndoCaver_MiTB0_GAM_DSA_LoCoS_Kvasir.pt
```

## What the example does

The script trains EndoCaver to predict:

- restored RGB image from a degraded input
- binary segmentation mask

The loss is:

```text
LoCoS-weighted L1 restoration loss + segmentation BCE/Dice loss
```

The LoCoS weighting in this repository is a small public version: early epochs give more weight to deblurring, then the emphasis moves toward segmentation.

## Inference after training

```bash
python scripts/infer.py \
  --images /path/to/test/images \
  --checkpoint checkpoints/EndoCaver_MiTB0_GAM_DSA_LoCoS_Kvasir.pt \
  --out outputs/kvasir_demo
```

## Notes

- The example is written for Kvasir-SEG only.
- Use 224 x 224 input resolution unless you adjust the decoder design.
- If GPU memory is limited, reduce `--batch-size`.
- The script saves the checkpoint with the best validation Dice.
