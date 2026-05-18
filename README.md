# photonic-crystal-gan

GAN-based image generation and enhancement on a custom photonic crystal dataset using PyTorch. Explores progression from Vanilla GAN to a two-stage DCGAN architecture based on the paper *"DCGAN and Photonic Power Dividers"* (Journal of Applied Physics, March 2025).

---

## Background

Photonic crystals are periodic nanostructures that control the flow of light. This project applies generative adversarial networks to a small custom dataset of 22 binary geometric microstructure images, exploring whether GANs can learn and reproduce their structural patterns — and ultimately generate and enhance new ones.

---

## Notebooks

| Notebook | Description | Status |
|---|---|---|
| Vanilla_GAN_Photonic_Crystal | Fully connected GAN baseline on 22 photonic crystal images | ✅ Done |
| DCGAN_20x20 | Stage 1 — Convolutional GAN generating 20×20 photonic crystal images | ✅ Done  |
| DCGAN_200x200 | Stage 2 — Convolutional GAN enhancing 20×20 outputs to 200×200 | 🔄 Coming soon |

---

## Two-Stage DCGAN Architecture

Based on the two-stage DCGAN methodology described in:

> Sengor, C. N., Ay, F., & Perkgoz, C. (2025). *A deep learning approach for 
> high-resolution and enhanced efficiency in photonic power dividers.* 
> Journal of Applied Physics, 137, 124903. https://doi.org/10.1063/5.0255080

PyTorch implementation of the two-stage DCGAN methodology described in the paper above, 
developed in collaboration with the physics department. Original paper code in TensorFlow/Keras.

**Stage 1** — Generates low-resolution 20×20 photonic crystal images from noise.

**Stage 2** — Takes Stage 1 outputs upscaled to 200×200 via pixel replication, then trains a second DCGAN to produce high-resolution enhanced images.

This pipeline addresses both generation and quality enhancement in a single progression.

---

## Stack

Python, PyTorch, torchvision, TensorBoard, PIL, Matplotlib

---

## Dataset

22 photonic crystal images (custom, not publicly available). Resized and normalized to [-1, 1] for training.

---

## Results

### Vanilla GAN — Baseline (Mode Collapse Expected)

Fully connected architecture with no spatial awareness. Mode collapse observed — G converged to near-identical outputs across all fixed noise vectors. Loss D → ~0.005, Loss G → ~8.5 by epoch 1000.

This confirms that fully connected layers cannot capture the geometric structure of photonic crystal images — motivating the DCGAN approach.

### Stage 1 DCGAN — 20×20 Generation
Losses converged to Nash equilibrium (~0.693) around epoch 500, stable through 2500 epochs. Binary output shows learned silicon-air structural patterns with some mode collapse expected from the 22-image dataset constraint.

→ See `DCGAN_20x20/` for full results, architecture details, and generated images.

### Stage 2 DCGAN — 200×200 Enhancement
Coming soon.
