# Runtime and exported artifacts

This release includes paper-aligned checkpoint and ONNX artifact names:

```text
checkpoints/EndoCaver_MiTB0_GAM_DSA_LoCoS_Kvasir.pt
checkpoints/EndoCaver_MiTB0_GAM_DSA_LoCoS_Kvasir.onnx
```

The name follows the arXiv description:

- EndoCaver
- MiT-B0 encoder
- Global Attention Module (GAM)
- Deblurring-Segmentation Aligner (DSA)
- LoCoS-style multi-task optimisation
- Kvasir reference weights

## Export ONNX

```bash
python scripts/export_onnx.py \
  --local-files-only \
  --checkpoint checkpoints/EndoCaver_MiTB0_GAM_DSA_LoCoS_Kvasir.pt \
  --out checkpoints/EndoCaver_MiTB0_GAM_DSA_LoCoS_Kvasir.onnx
```

ONNX I/O names:

```text
input:  image
outputs: deblur_output, segmentation_output
```

## Runtime checks run for this release

Evaluation was run on `samples/kvasir/images/kvasir_sample_01.png`.

| mode | provider/device | avg latency | fps |
| --- | --- | ---: | ---: |
| PyTorch checkpoint | cuda:0 | 36.22 ms | 27.61 |
| ONNX Runtime CPU | CPUExecutionProvider | 162.90 ms | 6.14 |
| ONNX Runtime GPU | CUDAExecutionProvider | 11.41 ms | 87.65 |

Full raw output is stored in `docs/runtime_eval.json`.

Commands:

```bash
python scripts/eval_runtime.py --mode pt --local-files-only --device cuda:0 --iters 5 --warmup 2
python scripts/eval_runtime.py --mode onnx-cpu --iters 3 --warmup 1
python scripts/eval_runtime.py --mode onnx-gpu --iters 5 --warmup 2
```

Note: ONNX GPU evaluation requires an ONNX Runtime build with `CUDAExecutionProvider`, e.g. `onnxruntime-gpu`.
