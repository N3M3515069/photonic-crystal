# Vanilla GAN — Photonic Crystal

A fully connected GAN baseline trained on 22 custom photonic crystal images using PyTorch.

Built as Stage 1 of the progression from Vanilla GAN → DCGAN on a real research dataset in collaboration with the physics department.

---

## Background

Photonic crystals are periodic nanostructures that control the flow of light. This project applies a Vanilla GAN to a small custom dataset of 22 binary geometric microstructure images to explore whether a fully connected architecture can learn and reproduce their structural patterns.

---

## Results

### Mode Collapse — Final Output
G collapsed to producing near-identical outputs across all 22 fixed noise vectors by the final epoch. All generated images converged to the same averaged texture with no structural variation.

![TensorBoard Output](assets/tensorboard.png)

**Loss D: ~0.005 — Loss G: ~8.5** — D completely dominated, G stopped exploring.

This is expected behavior for a fully connected GAN on 22 images with no spatial awareness.

---

## Why This Fails and Why DCGAN is Next

| Problem | Cause | DCGAN Fix |
|---|---|---|
| Mode collapse | No BatchNorm, D dominates immediately | BatchNorm stabilizes adversarial balance |
| No spatial understanding | Linear layers flatten pixel relationships | Conv layers preserve spatial structure |
| Dataset too small for FC | 22 images, no regularization | Conv layers generalize better with less data |
| All outputs identical | G memorizes one output | Strided conv + BatchNorm prevents this |

Vanilla GAN treats a 64×64×3 image as a flat 12,288-dim vector — all spatial relationships between pixels are destroyed. Photonic crystal images have precise geometric structure (edges, patterns, binary shapes) that only convolutional layers can capture.

→ See `DCGAN_Photonic_Crystal` notebook for the improved implementation.

---

## Architecture

### Discriminator
Input: `12288` (64×64×3 flattened)

```
Linear(12288 → 512) → LeakyReLU(0.2)
Linear(512 → 256)   → LeakyReLU(0.2)
Linear(256 → 1)     → Sigmoid
```

### Generator
Input: `z_dim = 100` (random noise vector)

```
Linear(100 → 256)    → LeakyReLU(0.2)
Linear(256 → 512)    → LeakyReLU(0.2)
Linear(512 → 12288)  → Tanh
Output reshaped to (3, 64, 64)
```

---

## Training

| Hyperparameter | Value |
|---|---|
| Epochs | 1000 |
| Batch size | 22 (full dataset) |
| Noise dim (z) | 100 |
| Learning rate | 0.0002 |
| Optimizer | Adam |
| Loss function | BCELoss |
| Image size | 64×64 |

---

## Dataset

22 photonic crystal images (custom, not publicly available). Resized to 64×64 and normalized to [-1, 1].

---

## Stack

Python, PyTorch, torchvision, TensorBoard, PIL, Matplotlib
