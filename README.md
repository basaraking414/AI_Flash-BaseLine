# DC-LIDM: Latent Illumination Decomposition with Dual Reconstruction Constraints for Night-Time Flash Portrait Enhancement

**English** | [中文](README_zh-CN.md)

Official code release for:

> Liqing Wang, Ming Ronnier Luo, *"Latent Illumination Decomposition with Dual Reconstruction
> Constraints for Night-Time Flash Portrait Enhancement"*, IEEE Access (under review).

DC-LIDM decomposes a low-light portrait into **reflectance `R`**, a **global environmental
illumination `L_env`** and a **spatially varying flash map `L_flash`**, recombines them in
**linear RGB** as `I = R × (L_env + α·L_flash)`, and predicts the flash map with **mask-guided
cross-attention** (the portrait mask acts as the query). Two reconstruction constraints
(input side and ground-truth side) share the same reflectance and regularize the decomposition.
A single scalar `α` gives continuous control of the flash strength at inference time.

**Main model:** Restormer backbone + DC-LIDM, 1.54 M parameters, 8.32 G FLOPs @ 256×256,
**PSNR 31.35 / ΔE 2.77 / SSIM 0.968** on the 50-image validation split (paper Table I, "Ours").

---

## 1. Scope: what is (and is not) in this repository

| Item | Included | Note |
|---|---|---|
| Main model `Restormer_AIFlash_mask_attention` | ✅ | `basicsr/models/archs/my_restormer_arch.py` |
| Standalone DC-LIDM module (`DC_LIDM.py`, plug-and-play interface) | ✅ | for inserting into other encoder–decoder backbones |
| Losses / metrics (Charbonnier, FaceLab, perceptual, gradient, TV; PSNR/SSIM/ΔE) | ✅ | `basicsr/models/losses/`, `basicsr/metrics/` |
| Training / evaluation / inference entry points | ✅ | `basicsr/my_train.py`, `basicsr/test.py`, `inference.py` |
| Pre-trained main model | ✅ (GitHub Releases) | `best_Restormer_LIDM.pth`, see §5 |
| Full-scale training data (550 pairs) | ❌ | public challenge data, get it from the challenge website (§4) |
| The 7 external comparison methods of Table I | ❌ | RetinexNet / SCI / KinD / Zero-DCE / RetinexFormer / NAFNet / Uformer / plain Restormer: please adapt them from their official repositories |
| Cross-backbone variants (NAFNet+DC-LIDM, Uformer+DC-LIDM) | ❌ | adapt `DC_LIDM.py` into those backbones from their official code |
| Ablation variants V1/V2/V3 | ❌ | historical intermediate models, not released |

**Comparison protocol** used in the paper (and reproducible with the code here): same 550/50 split,
same input pipeline, one unified supervised loss, `crop_border=0`, no Y-channel conversion, and
**ΔE averaged inside the portrait mask** (CIE76 in CIELAB).

---

## 2. Verified end-to-end (measured in this repository)

Hardware / software: **NVIDIA RTX 4060 Laptop GPU (8 GB), CUDA 12.6,
torch 2.10.0+cu126, torchvision 0.25.0+cu126, Python 3.13.11** (see `requirements.txt`).

| Stage | Command | Measured result |
|---|---|---|
| **Training** (smoke: 1 sample, 20 iters) | `python -m basicsr.my_train -opt configs/smoke_val1.yaml` | exit 0, 23 s; all 6 losses computed; validation at iter 10 and 20; writes `best_net_g_psnr.pth`, `best_net_g_deltaE.pth`, `net_g_latest.pth`, `.state` files, loss curves, TensorBoard events and R/E/F/illumination visualizations. A second run correctly **auto-resumed** from `training_states/` |
| **Evaluation** (released weight, 50 val images) | `python -m basicsr.test -opt configs/eval_val50.yaml` | exit 0, 48 s → **PSNR 31.3518 / ΔE 2.7683 / SSIM 0.9676** — reproduces paper Table I "Ours" (31.35 / 2.77 / 0.968) |
| **Inference** (batch) | `python inference.py` | 10/10 images written to `result/test/`, exit 0 |
| **Inference** (released weight) | `python inference.py --config configs/my_aiflash.yaml --weights best_model/best_Restormer_LIDM.pth` | 10/10 at `α=1`, 10/10 at `α=0`, exit 0 |
| **Tiled inference** | `python inference.py --tile 512 --tile_overlap 32` | mean abs. difference vs full-resolution: **1.62/255** (max 21/255, seam blending only) |
| **Flash control `α`** | `python inference.py --alpha 0 / 0.5 / 1 / 2` | mean luminance rises monotonically: 0.527 → 0.568 → 0.602 → 0.660 (locally trained smoke checkpoint) |
| **Input-side constraint** | released weight, `--alpha 0` | `mean|α=0 − input| = 3.36/255` (1.3 %): the decomposition does reconstruct the low-light input, as Eq. (6) requires |
| **CPU inference** | `python inference.py --device cpu` | 10.8 s per 1024×768 image |

> The smoke model itself is only 20 iterations old, so its PSNR/ΔE values are meaningless as quality
> numbers — they only demonstrate that the whole training/validation/evaluation plumbing runs.

---

## 3. Repository structure

```
AI-Flash-project/
├── basicsr/                              # trimmed BasicSR framework
│   ├── __init__.py
│   ├── data/                             # Dataset_MaskedImage / Dataset_PairedImage (with masks)
│   ├── models/
│   │   ├── archs/
│   │   │   ├── my_restormer_arch.py       # main model + EnvLightHead + FlashHeadMasked + sRGB<->linear
│   │   │   └── DC_LIDM.py                 # standalone DC-LIDM module (plug-and-play)
│   │   ├── losses/losses.py               # Charbonnier / Charbonnier_mask / FaceLab / Perceptual / Gradient / TVSmooth
│   │   ├── my_ai_flash_model.py           # AIFlashModel: dual constraints, loss assembly, best checkpoints
│   │   ├── image_restoration_model.py     # generic framework model
│   │   ├── base_model.py  lr_scheduler.py
│   ├── metrics/psnr_ssim.py               # PSNR / SSIM / ΔE (masked CIE76)
│   ├── utils/
│   ├── my_train.py                        # training entry point (mask-aware) -- use this one
│   ├── test.py                            # evaluation entry point
│   ├── train.py                           # original entry point without mask support
│   └── visualize.py                       # intermediate-tensor and loss-curve plotting
├── configs/
│   ├── my_aiflash.yaml                    # paper training config (Table I "Ours" / Table II "Full")
│   ├── eval_val50.yaml                    # reproduce Table I "Ours" on val_50
│   ├── smoke_val1.yaml                    # 1-sample / 20-iteration smoke test
│   └── inference.yaml                     # inference / efficiency template
├── best_model/                            # put released weights here (git-ignored)
├── inference.py                           # single-image and batch inference, --alpha control
├── requirements.txt  setup.py  setup.cfg  VERSION
├── LICENSE  THIRD_PARTY_NOTICES.md
└── README.md  README_zh-CN.md  readme.txt
```

---

## 4. Data

Dataset: **NTIRE 2026 AI Flash Portrait** (Track 3 of the 3rd Restore Any Image Model Challenge),
download from <https://www.codabench.org/competitions/12885/>.
This repository does **not** redistribute the data — please follow the challenge terms.

The paper uses the 600 public pairs split into **550 train / 50 validation**:

```
datasets/
├── train_550/{gt,input,mask_personmask}/
└── val_50/{gt,input,mask_personmask}/      # all reported metrics

datasets/val_1/{gt,input,mask_personmask}/  # 1 sample, used by the smoke config
dataset/test/{input,mask_personmask}/       # default input of inference.py
result/test/                                # default output of inference.py
```

Masks are paired with images **by file name** (`input/001.jpg` ↔ `mask_personmask/001.jpg`) and
binarized with `mask > 0`, matching the training pipeline. Images are rounded/cropped to multiples
of 8 as required by the multi-scale backbone.

---

## 5. Weights

Only the main model is published, as a GitHub **Release** asset:

| File | Architecture | Reproduces |
|---|---|---|
| `best_Restormer_LIDM.pth` | `Restormer_AIFlash_mask_attention` | PSNR 31.35 / ΔE 2.77 / SSIM 0.968 (Table I "Ours") |

1. Download it from the repository's **Releases** page.
2. Put it at `best_model/best_Restormer_LIDM.pth` (the directory is git-ignored, create it if needed).
3. Optional integrity check: `sha256sum best_model/best_Restormer_LIDM.pth` /
   `Get-FileHash best_model/best_Restormer_LIDM.pth -Algorithm SHA256`.

The checkpoint is stored as `torch.save({'params': state_dict})`; the loader also accepts
`params_ema`, `state_dict` and single-key nested archives, and strips `module.` prefixes.

---

## 6. Installation

```bash
# Python 3.13
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# CUDA build of PyTorch (example: CUDA 12.6)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
```

Run all commands **from the repository root**:

```bash
pip install -e .                          # optional; enables `python basicsr/my_train.py ...`
python -m basicsr.my_train -opt configs/my_aiflash.yaml     # always works
```

`python -m ...` puts the repository root on the import path, so `import basicsr` resolves to the
code in this repository even if another editable `basicsr` installation exists in the environment.

---

## 7. Training

```bash
# Full paper training (300k iterations, 550 pairs) -- configs/my_aiflash.yaml
python -m basicsr.my_train -opt configs/my_aiflash.yaml

# Smoke test on one sample (20 iterations, ~25 s) -- verifies the whole chain
python -m basicsr.my_train -opt configs/smoke_val1.yaml
```

Config fields correspond to the paper as follows:

| Paper | Config field |
|---|---|
| AdamW, β=(0.9, 0.999), wd 1e-4, lr 3e-4 cosine-annealed | `train.optim_g`, `train.scheduler` |
| batch size 1, 256×256 patches, single 8 GB GPU | `datasets.train.batch_size_per_gpu`, `gt_size` |
| α = 1 during training | `train.alpha_sampling: false` |
| λ₁ `L_rec` = 2.0 | `train.pixel_opt` (`CharbonnierLoss_mask`) |
| λ₂ `L_input` = 1.0 | `train.input_recon_opt` (`CharbonnierLoss`) |
| λ₃ `L_Lab` = 1.0 (λ_L 1.5, λ_ab 1.0) | `train.luminance_opt` (`FaceLabLoss`) |
| λ₄ `L_percept` = 0.001 (conv3_4, conv4_4) | `train.perceptual_opt` |
| λ₅ `L_grad` = 0.2 | `train.grad_opt` |
| additional TV regularizer on `L_env` (not listed in Eq. 7) | `train.smooth_opt` (`TVSmoothLoss`, 0.1) |

Outputs: `experiments/<name>/` with `models/` (`best_net_g_psnr.pth`, `best_net_g_deltaE.pth`,
`net_g_latest.pth`, `best_metrics.json`), `training_states/` (`.state` + loss history),
`loss_curves/`, `visualization/`, `val_visualizations/`, and a full training log.
TensorBoard logs go to `tb_logger/<name>/`. Training resumes automatically from the largest
`.state` file if `experiments/<name>/training_states/` already contains one.

---

## 8. Evaluation (reproduces Table I, "Ours")

```bash
# expects best_model/best_Restormer_LIDM.pth and datasets/val_50/
python -m basicsr.test -opt configs/eval_val50.yaml
```

Expected log line (measured: PSNR 31.3518 / ΔE 2.7683 / SSIM 0.9676 in 48 s):

```
INFO: Validation ValSet50,   # psnr: 31.3518   # deltaE: 2.7683   # ssim: 0.9676
```

---

## 9. Inference

```bash
# default: configs/my_aiflash.yaml + best_model/best_Restormer_LIDM.pth
#          dataset/test/{input,mask_personmask} -> result/test/
python inference.py

python inference.py --config configs/my_aiflash.yaml \
                    --weights best_model/best_Restormer_LIDM.pth \
                    --input_dir dataset/test/input \
                    --mask_dir dataset/test/mask_personmask \
                    --output_dir result/test

# controllable flash strength alpha (paper Eq. 1 / Fig. 12)
python inference.py --alpha 0     --output_dir result/alpha0.0   # ~= reconstructed low-light input
python inference.py --alpha 0.5   --output_dir result/alpha0.5
python inference.py --alpha 1.5   --output_dir result/alpha1.5
python inference.py --alpha 2.0   --output_dir result/alpha2.0

# low-memory tiled inference (tile must be a multiple of 8)
python inference.py --tile 512 --tile_overlap 32

# single image / CPU / specific GPU
python inference.py --input_dir dataset/test/input/001.jpg \
                    --mask_dir dataset/test/mask_personmask/001.jpg
python inference.py --device cpu
python inference.py --device cuda:0
```

| Option | Default | Meaning |
|---|---|---|
| `--config` | `configs/my_aiflash.yaml` | only its `network_g` section is used |
| `--weights` | `best_model/best_Restormer_LIDM.pth` | checkpoint path |
| `--input_dir` | `dataset/test/input` | directory or single image |
| `--mask_dir` | `dataset/test/mask_personmask` | directory or single mask |
| `--output_dir` | `result/test` | created automatically, PNG output |
| `--alpha` | `1.0` | flash strength, paper Eq. (1) |
| `--tile` / `--tile_overlap` | `None` / `32` | tiled inference; full resolution otherwise |
| `--device` | `auto` | `auto` / `cuda` / `cuda:0` / `cpu`, falls back to CPU automatically |

Images are padded to a multiple of 8, the output is cropped back and clamped to [0, 1]. Processing
is logged per image; the script exits with code 1 if every image failed.

---

## 10. Paper ↔ code correspondence

| Paper | Code |
|---|---|
| Eq. (1) `I = R × (L_env + α·L_flash)` | `Restormer_AIFlash_mask_attention.forward`: `illumination = env_light_img + alpha * flash_map` |
| Eq. (2) environmental illumination | `EnvLightHead` + `0.9·σ(·)` and bilinear upsampling (see implementation notes) |
| Eq. (3) mask-as-query cross-attention | `FlashHeadMasked.forward` (mask → Q, latent → K/V) |
| Eq. (4) gated fusion `(1-g)⊙F + g⊙F̂` | `FlashHeadMasked` `gate_conv` (two conv layers + sigmoid gate) |
| Eq. (5) `L_flash = σ(D(F')) ⊙ (γM + 1 − γ)`, γ = 0.9 | `soft_mask = 0.9 * mask + 0.1` |
| Eq. (6) dual reconstruction constraints | `my_ai_flash_model.optimize_parameters`: `input_recon = R × L_env` vs `srgb_to_linear(lq)`; GT side enforced by `pixel_opt` |
| Eq. (7) loss weighting | `configs/my_aiflash.yaml` → `train.*_opt.loss_weight` |
| linear RGB processing | `srgb_to_linear` / `linear_to_srgb` |
| ΔE (masked CIE76) | `basicsr/metrics/psnr_ssim.py: calculate_deltaE(img1, img2, mask)` |
| loss implementations | `basicsr/models/losses/losses.py` |

**Implementation notes**

- `L_env` is implemented as a **low-resolution spatial map** `(B, 3, H/16, W/16)` produced by a
  convolutional head with Gaussian smoothing, scaled by `0.9·σ(·)` and bilinearly upsampled to the
  output resolution; its low-frequency character is encouraged by `TVSmoothLoss` (`train.smooth_opt`).
  Consequently the training objective contains **six** terms, while paper Eq. (7) lists five.
- Validation and evaluation always run with `α = 1`.
- Every config uses `use_image: false`, i.e. metrics are computed on tensors with the portrait mask
  (this is the protocol behind the reported ΔE). The `use_image: true` branch is not supported by
  the mask-aware metric signatures.

---

## 11. Known issues

1. `basicsr/train.py` does not pass the portrait mask, so it cannot train `AIFlashModel`
   (`KeyError: 'mask'`). Use `basicsr/my_train.py`.
2. `pad_test()` pads the low-light input but not the mask; this only matters for inputs whose
   height/width are not multiples of 8 (the dataset already rounds sizes to multiples of 8).
3. `basicsr/metrics/fid.py` imports a module that is not part of this repository (FID is unused).
4. `Dataset_GaussianDenoising` contains an incorrect import path
   (`from basicsr.utils.scandir import scandir`); the class is unused here.
5. `python basicsr/my_train.py ...` only works after `pip install -e .`; use
   `python -m basicsr.my_train ...` otherwise.
6. The external comparison methods and the cross-backbone variants are intentionally not included
   (see §1), so Tables I (other rows), III and IV (baseline columns) cannot be reproduced from this
   repository alone.

---

## 12. License and citation

Released under the **Apache License 2.0** (`LICENSE`). The `basicsr/` directory is derived from
[BasicSR](https://github.com/XPixelGroup/BasicSR) (Apache-2.0, © Xintao Wang); the main model
follows the Restormer architecture. See `THIRD_PARTY_NOTICES.md` for details.

```bibtex
@article{wang2026dclidm,
  title   = {Latent Illumination Decomposition with Dual Reconstruction Constraints
             for Night-Time Flash Portrait Enhancement},
  author  = {Wang, Liqing and Luo, Ming Ronnier},
  journal = {IEEE Access},
  year    = {2026},
  note    = {under review}
}
```

If you use the NTIRE 2026 AI Flash Portrait dataset, please also cite the challenge report
(`guan2026` in the paper's bibliography) and follow its data terms.