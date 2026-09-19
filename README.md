# DC-LIDM: Night-Time Flash Portrait Enhancement

**English** | [中文](README_zh-CN.md)

Official code for *"Latent Illumination Decomposition with Dual Reconstruction Constraints for
Night-Time Flash Portrait Enhancement"* — Liqing Wang, Ming Ronnier Luo (IEEE Access, under review).

DC-LIDM decomposes a low-light portrait into reflectance `R`, ambient light `L_env` and a spatially
varying flash map `L_flash`, recombines them in linear RGB as `I = R × (L_env + α·L_flash)`, and
estimates the flash map with mask-guided cross-attention (the portrait mask acts as the query).
The scalar `α` controls flash strength at inference time.

**val_50 (50 images): PSNR 31.35 · ΔE 2.77 · SSIM 0.968** — 1.54 M params, 8.32 G FLOPs @ 256×256.

## Contents

| Path | What it is |
|---|---|
| `basicsr/models/archs/my_restormer_arch.py` | main model (Restormer + DC-LIDM) |
| `basicsr/models/archs/DC_LIDM.py` | standalone DC-LIDM module (plug-and-play) |
| `basicsr/models/losses/losses.py` | Charbonnier / FaceLab / perceptual / gradient / TV losses |
| `basicsr/metrics/psnr_ssim.py` | PSNR / SSIM / masked ΔE |
| `basicsr/my_train.py` · `basicsr/test.py` | training · evaluation |
| `inference.py` | inference, with `--alpha` flash control |
| `configs/my_aiflash.yaml` · `configs/inference.yaml` | training config · inference template |

Not included: the external comparison methods of Table I (RetinexNet, SCI, KinD, Zero-DCE,
RetinexFormer, NAFNet, Uformer, plain Restormer). Please adapt them from their official repositories.

## Setup

Python 3.13:

```bash
python -m venv venv
venv\Scripts\activate                                # Windows
pip install -r requirements.txt
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
```

Run all commands from the repository root.

## Data

NTIRE 2026 AI Flash Portrait (Track 3 of the RAIM Challenge):
<https://www.codabench.org/competitions/12885/> — the data is public and is **not** redistributed here.

```
datasets/train_550/{gt,input,mask_personmask}/   # 550 training pairs
datasets/val_50/{gt,input,mask_personmask}/      # 50 validation pairs (all reported metrics)
```

Masks are matched to images by file name (`input/001.jpg` ↔ `mask_personmask/001.jpg`) and
binarized with `mask > 0`.

## Weights

Download `best_Restormer_LIDM.pth` from the **Releases** page and put it at
`best_model/best_Restormer_LIDM.pth`. It reproduces the numbers above.

## Usage

```bash
# training (configs/my_aiflash.yaml: 550 pairs, 300k iterations)
python -m basicsr.my_train -opt configs/my_aiflash.yaml

# evaluation: set path.pretrain_network_g to the released weight first
# (keep only the `val` dataset in the config to evaluate the validation split alone)
python -m basicsr.test -opt configs/my_aiflash.yaml

# inference (defaults: configs/my_aiflash.yaml + best_model/best_Restormer_LIDM.pth)
python inference.py --input_dir datasets/val_50/input \
                    --mask_dir  datasets/val_50/mask_personmask \
                    --output_dir result/val_50

# controllable flash strength alpha (0 ~= reconstructed low-light input, 1 = training setting)
python inference.py --alpha 0.5 --output_dir result/alpha0.5

# tiled inference for low-memory GPUs (tile must be a multiple of 8); CPU also works
python inference.py --tile 512
```

Inference options: `--config`, `--weights`, `--input_dir`, `--mask_dir`, `--output_dir`, `--alpha`,
`--tile` / `--tile_overlap`, `--device` (`auto` / `cuda` / `cuda:0` / `cpu`).

## Notes

- Use `basicsr/my_train.py` for training: `basicsr/train.py` does not pass the portrait mask.
- Metrics are computed on the network output together with the portrait mask (`use_image: false`) —
  this is the protocol behind the reported ΔE. Validation always runs with `α = 1`.
- `L_env` is implemented as a low-resolution spatial map regularized by a TV term, so the training
  objective has six terms (paper Eq. 7 lists five).
- `python basicsr/my_train.py ...` needs `pip install -e .`; `python -m basicsr.my_train ...`
  works out of the box.

## Citation

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

## License

Apache License 2.0 — see `LICENSE`. The `basicsr/` framework is derived from
[BasicSR](https://github.com/XPixelGroup/BasicSR); see `THIRD_PARTY_NOTICES.md`.