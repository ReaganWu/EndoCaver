# EndoCaver

This repository provides the **official reference implementation** of **EndoCaver**, a lightweight dual-decoder transformer framework accepted by **IEEE ICASSP 2026**.

> **EndoCaver: Handling Fog, Blur and Glare in Endoscopic Images via Joint Deblurring–Segmentation**  
> Zhuoyu Wu, Wenhui Ou, Pei-Sze Tan, Jiayan Yang, Wenqi Fang, Zheng Wang, Raphaël C.-W. Phan  
> *Accepted by IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026*

---

## Overview

Endoscopic image analysis in real-world clinical environments is frequently affected by **lens fogging, motion blur, and specular highlights**, which significantly degrade the reliability of automated polyp segmentation systems.

**EndoCaver** addresses this challenge through a **joint deblurring–segmentation framework** that is explicitly designed for **robustness under image degradation** and **resource-constrained clinical deployment**.

The core design philosophy of EndoCaver is to:
- Preserve **task synergy** between restoration and segmentation,
- Avoid excessive architectural complexity,
- Maintain **strong parameter efficiency** without sacrificing performance.

---

## Key Contributions

- **Unidirectional-Guided Dual-Decoder Architecture**  
  A lightweight dual-decoder design where deblurring cues are explicitly transferred to the segmentation pathway, avoiding bidirectional interference and unnecessary redundancy.

- **Global Attention Module (GAM)**  
  A cross-scale aggregation mechanism that unifies multi-level encoder features into a global representation, enhancing structural awareness with minimal computational overhead.

- **Deblurring–Segmentation Aligner (DSA)**  
  A cross-attention based alignment module that injects restoration priors into segmentation, improving boundary integrity and robustness under severe degradations.

- **LoCoS: Cosine-Annealed Multi-Task Optimization**  
  A task-adaptive loss weighting strategy that stabilizes joint optimization by progressively shifting focus from deblurring to segmentation.

---

## Method Summary

Given an input endoscopic image affected by real-world degradations, EndoCaver jointly predicts:
- a restored (deblurred) image, and
- a corresponding segmentation mask,

within a **single unified network**.

The framework consists of:
- a lightweight hierarchical transformer encoder,
- a Global Attention Module for cross-scale enhancement,
- a **deblurring decoder** responsible for restoration,
- a **segmentation decoder** guided by aligned restoration features.

This unidirectional guidance enables effective information transfer while maintaining architectural simplicity.

---

## Efficiency and Robustness

EndoCaver is explicitly optimized for clinical deployment scenarios:

- **Parameters**: 7.8M  
- **Computational Complexity**: 11.9 GMACs  
- **Performance**:
  - Dice 0.922 on clean Kvasir-SEG
  - Dice 0.889 under severe synthetic degradations

Despite its compact size, EndoCaver demonstrates superior robustness compared to substantially larger state-of-the-art models.

---

## Code Availability

This repository is released as a **research reference** to support transparency and reproducibility of the proposed method.

- The code focuses on **core architectural components** (GAM, DSA, LoCoS).
- Training pipelines, datasets, and deployment scripts are **not provided** in this repository.
- The implementation is intended for **method-level understanding and academic reference**, rather than direct end-to-end reproduction.

---

## Paper and Citation

A preprint version of the paper will be available on **arXiv**.

If you find this work useful, please consider citing:

```bibtex
@inproceedings{wu2026endocaver,
  title={EndoCaver: Handling Fog, Blur and Glare in Endoscopic Images via Joint Deblurring--Segmentation},
  author={Wu, Zhuoyu and Ou, Wenhui and Tan, Pei-Sze and Yang, Jiayan and Fang, Wenqi and Wang, Zheng and Phan, Rapha{\"e}l C.-W.},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
