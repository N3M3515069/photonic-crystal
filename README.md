# photonic-crystal-gan

GAN-based image generation on a custom photonic crystal dataset using PyTorch. Explores progression from Vanilla GAN to DCGAN for generating and enhancing binary geometric microstructure images.

## Background

Photonic crystals are periodic nanostructures that control the flow of light. This project applies generative adversarial networks to a small custom dataset of photonic crystal images, exploring whether GANs can learn and reproduce their structural patterns.

## Notebooks

| Notebook | Description |
|---|---|
| Vanilla_GAN_Photonic_Crystal | Fully connected GAN baseline on 22 photonic crystal images |
| DCGAN_Photonic_Crystal | Convolutional GAN with improved spatial feature learning *(coming soon)* |

## Stack

Python, PyTorch, torchvision, TensorBoard, PIL

## Dataset

22 photonic crystal images (custom, not publicly available). Images resized to 64×64 and normalized to [-1, 1] for training.

## Results

### Vanilla GAN — Final Epoch Output
![TensorBoard Output](tensorboard.png)

> Fake images (top) show partial structural similarity to real photonic crystal images (bottom).
> DCGAN results coming soon for comparison.


