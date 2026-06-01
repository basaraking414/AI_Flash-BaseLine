# AI Flash Portrait

> 基于 Retinex 分解的人像闪光重光照系统 | A Retinex-based Portrait Flash Relighting System

A deep learning system that generates high-quality flash portraits from non-flash input images. Built on the BasicSR framework with a Restormer-based encoder-decoder architecture, the model decomposes images into reflectance and illumination using Retinex theory, then reconstructs flash-lit output via mask-guided cross-attention.

**Core formula**: `output = R × (E + α × F)` in linear RGB space

- **R** — Reflectance (scene albedo, illumination-invariant)
- **E** — Environmental light (ambient illumination, spatially smooth)
- **F** — Flash light (estimated via cross-attention with person mask)
- **α** — Flash intensity control (default 1.0)

---

## Architecture

```
Input (sRGB) ──→ srgb_to_linear ──→ PatchEmbed (3×3 Conv)
                                          │
                    ┌─────────────────────┤
                    │     Encoder         │
                    │  Level 1: dim=16    │
                    │  Level 2: dim=32    │
                    │  Level 3: dim=64    │
                    │  Latent: dim=128    │
                    │                     │
                    │   ┌─────────────────┤
                    │   │                 │
                    │   ▼                 ▼
                    │  EnvLightHead    FlashHeadMasked
                    │  (Conv+Gauss     (Cross-Attention:
                    │   blur, cap      Mask→Q, Latent→K,V)
                    │   at 0.9)        + Gated Fusion
                    │   │                 │
                    │   ▼                 ▼
                    │  E (env light)   F (flash map)
                    │   │                 │
                    │   └────┬────────────┘
                    │        │
                    │        ▼
                    │  illumination = E + α × F
                    │        │
                    ├────────┤
                    │     Decoder
                    │  Level 3→2→1
                    │  + Skip connections
                    │  + Refinement
                    │        │
                    │        ▼
                    │  Output Conv → sigmoid → R (reflectance)
                    │        │
                    └────────┤
                             │
                             ▼
                    output = R × illumination
                             │
                             ▼
                    linear_to_srgb → Final Output (sRGB)
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| `Restormer_AIFlash_mask_attention` | `basicsr/models/archs/my_restormer_arch.py` | Main network: U-Net encoder-decoder with Restormer TransformerBlocks |
| `EnvLightHead` | `my_restormer_arch.py:192` | Ambient light estimation with Gaussian blur for spatial smoothness |
| `FlashHeadMasked` | `my_restormer_arch.py:361` | Cross-attention flash estimation (mask→Q, latent→K,V) with gated fusion |
| `AIFlashModel` | `basicsr/models/my_ai_flash_model.py` | Training model: loss orchestration, validation, visualization |

---

## Project Structure

```
AI_Flash-BaseLine/
├── configs/                              # Training configurations
│   ├── my_aiflash.yaml                  # R0+C variant (main config)
│   ├── my_aiflash_ablation.yaml         # Ablation study config
│   └── flash.yaml                       # Baseline Restormer config
├── basicsr/
│   ├── models/
│   │   ├── archs/
│   │   │   ├── my_restormer_arch.py     # Main architecture (AIFlash)
│   │   │   ├── restormer_arch.py        # Original Restormer
│   │   │   ├── RetinexFormer_arch.py    # Comparison: RetinexFormer
│   │   │   ├── RetinexNet_arch.py       # Comparison: RetinexNet
│   │   │   ├── NAFNet_arch.py           # Comparison: NAFNet
│   │   │   ├── SwinIR_arch.py           # Comparison: SwinIR
│   │   │   ├── ZeroDCE_arch.py          # Comparison: ZeroDCE
│   │   │   └── KinD_arch.py             # Comparison: KinDNet
│   │   ├── losses/
│   │   │   └── losses.py               # 7 loss functions
│   │   └── my_ai_flash_model.py        # Training model (AIFlashModel)
│   ├── data/
│   │   └── my_mask_image_dataset.py    # Dataset with 5 mask types
│   ├── metrics/
│   │   └── psnr_ssim.py               # PSNR & DeltaE metrics
│   ├── my_train.py                     # Training script (main)
│   └── visualize.py                    # Visualization tools
├── datasets/                            # Data (not in repo)
│   ├── train_550/                       # Training set (550 images)
│   └── val_50/                          # Validation set (50 images)
├── experiments/                         # Experiment outputs
├── inference.py                         # Standalone inference script
└── tb_logger/                           # TensorBoard logs
```

---

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.x
- CUDA 11.8+

### Dependencies

```bash
pip install torch torchvision
pip install einops kornia natsort scikit-image tqdm pyyaml matplotlib opencv-python
```

### Setup

```bash
git clone <repo_url>
cd AI_Flash-BaseLine
```

---

## Dataset Preparation

### Directory Structure

```
datasets/
├── train_550/
│   ├── input/               # Non-flash input images
│   ├── gt/                  # Flash ground truth images
│   ├── mask_personmask/     # Person segmentation masks (primary)
│   ├── mask_hairmask/       # Hair region masks
│   ├── mask_skinmask/       # Skin region masks
│   ├── mask_skinmaskFace/   # Face skin masks
│   └── mask_skymask/        # Sky region masks
└── val_50/
    ├── input/
    ├── gt/
    └── mask_personmask/
```

### Mask Types

| Mask | Description | Usage |
|------|-------------|-------|
| `mask_personmask` | Full person body segmentation | Primary mask for FlashHeadMasked |
| `mask_hairmask` | Hair region | Loaded but not actively used |
| `mask_skinmask` | Skin region | Loaded but not actively used |
| `mask_skinmaskFace` | Face skin region | Loaded but not actively used |
| `mask_skymask` | Sky region | Loaded but not actively used |

> Note: Only `mask_personmask` is used in current training. Other masks are loaded and augmented but not passed to the loss functions.

---

## Usage

### Training

```bash
# Basic training
python -m basicsr.my_train -opt configs/my_aiflash.yaml

# Specify GPU
CUDA_VISIBLE_DEVICES=0 python -m basicsr.my_train -opt configs/my_aiflash.yaml

# Distributed training
python -m basicsr.my_train -opt configs/my_aiflash.yaml --launcher pytorch
```

Training features:
- **Auto-resume**: Detects the latest checkpoint in `experiments/{name}/training_states/`
- **Early stopping**: Patience=12 based on PSNR/DeltaE
- **Best model**: Saved as `best_net_g.pth` when metrics improve
- **Loss curves**: Plotted after each validation to `experiments/{name}/loss_curves/`
- **TensorBoard**: Logs to `tb_logger/`

### Inference

```bash
# Basic inference
python inference.py \
  --config configs/my_aiflash.yaml \
  --weights path/to/best_net_g.pth \
  --input_dir ./test/input \
  --mask_dir ./test/mask \
  --output_dir ./results

# With evaluation (requires GT)
python inference.py \
  --config configs/my_aiflash.yaml \
  --weights path/to/best_net_g.pth \
  --input_dir ./test/input \
  --mask_dir ./test/mask \
  --gt_dir ./test/gt \
  --output_dir ./results \
  --eval

# Tile mode for large images
python inference.py \
  --config configs/my_aiflash.yaml \
  --weights path/to/best_net_g.pth \
  --input_dir ./test/input \
  --mask_dir ./test/mask \
  --output_dir ./results \
  --tile 512
```

### Monitoring

```bash
# TensorBoard
tensorboard --logdir tb_logger/
```

---

## Configuration

### Key Parameters (`my_aiflash.yaml`)

| Section | Parameter | Value | Description |
|---------|-----------|-------|-------------|
| `network_g.type` | — | `Restormer_AIFlash_mask_attention` | Network architecture |
| `network_g.dim` | — | 16 | Base channel dimension |
| `network_g.num_blocks` | — | [2,2,2,2] | TransformerBlocks per encoder level |
| `network_g.heads` | — | [1,2,4,8] | Attention heads per level |
| `train.optim_g.type` | — | AdamW | Optimizer |
| `train.optim_g.lr` | — | 3e-4 | Learning rate |
| `train.total_iter` | — | 300,000 | Total training iterations |
| `train.alpha_sampling` | — | false | Fix alpha=1.0 (R0+C variant) |
| `train.use_grad_clip` | — | true | Gradient clipping (max_norm=0.01) |
| `val.val_freq` | — | 5000 | Validation frequency (iterations) |
| `val.window_size` | — | 8 | Padding window for validation |

### Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| `CharbonnierLoss_mask` | 2.0 | Background reconstruction (mask-aware Charbonnier) |
| `FaceLabLoss` | 1.0 | Face color fidelity in LAB color space |
| `GradientLoss` | 0.2 | Edge/texture preservation (Sobel-based) |
| `ColorLoss` | 0.3 | Low-frequency color consistency (patch-average MSE) |
| `PerceptualLoss` | 0.001 | Perceptual quality (VGG19 conv3_4, conv4_4) |
| `CharbonnierLoss` (input_recon) | 5.0 | Retinex constraint: R×E ≈ input (linear RGB) |
| `TVSmoothLoss` | 0.1 | Environmental light spatial smoothness (Total Variation) |

---

## Results

### Quantitative Comparison

| Method | PSNR ↑ | DeltaE ↓ | Description |
|--------|--------|----------|-------------|
| Restormer (baseline) | 30.21 | 3.33 | No Retinex decomposition |
| RetinexNet | 26.78 | 5.04 | Classic Retinex network |
| ZeroDCE | 28.00 | 4.17 | Zero-reference enhancement |
| NAFNet | 28.07 | 4.44 | Nonlinear Activation Free Network |
| KinDNet | 29.17 | 3.65 | Retinex-based decomposition |
| RetinexFormer | 30.73 | 3.08 | Transformer + Retinex |
| **AIFlash (Ours)** | **31.17** | **2.83** | **Retinex + mask attention (R0+C)** |

Best model: `experiments/R0C_input_recon_constraint_v4/models/best_net_g.pth` (285,000 iterations)

### Experiment Versions

| Version | Key Change | Score |
|---------|------------|-------|
| v3-2 | linear RGB, no mask attention | 82.77 |
| v4 | + mask attention | 82.72 |
| R0C v4 | + input reconstruction loss + TV smooth loss | Best PSNR/DeltaE |

---

## Known Issues

1. **Resume Training**: `model.resume_training()` is commented out in `my_train.py:180`. When resuming from a checkpoint, optimizer and scheduler states are NOT restored — only the iteration counter is recovered. This causes learning rate to reset to the initial value.

2. **Visualization Source Missing**: `basicsr/visualize.py` source file is deleted on the `old-iter-version` branch. Only the `.pyc` bytecode cache remains. If `__pycache__/` is cleared, training will crash with `ModuleNotFoundError`.

3. **Validation Image Limit**: Only the first 10 validation images are saved to disk during validation (`my_ai_flash_model.py:548`).

4. **PSNR Mask Not Applied**: The `calculate_psnr` function accepts a `mask` parameter but does not actually use it for filtering (the mask logic is commented out).

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{aiflash2026,
  title={AI Flash Portrait: Retinex-based Portrait Flash Relighting},
  author={Ricky Wang},
  year={2026}
}
```

---

## License

[Specify license]
