# DC-LIDM：夜间闪光人像增强

[English](README.md) | **中文**

论文 *"Latent Illumination Decomposition with Dual Reconstruction Constraints for Night-Time Flash
Portrait Enhancement"* 的官方代码 —— Liqing Wang, Ming Ronnier Luo（IEEE Access，审稿中）。

DC-LIDM 把低光人像分解为反射率 `R`、环境光 `L_env` 与空间变化的闪光图 `L_flash`，在线性 RGB 空间
按 `I = R × (L_env + α·L_flash)` 重组，并用掩码引导的交叉注意力（人像掩码作为 query）估计闪光图。
推理时用标量 `α` 连续控制闪光强度。

**val_50（50 张）：PSNR 31.35 · ΔE 2.77 · SSIM 0.968** —— 1.54 M 参数，256×256 输入 8.32 G FLOPs。

## 内容

| 路径 | 说明 |
|---|---|
| `basicsr/models/archs/my_restormer_arch.py` | 主模型（Restormer + DC-LIDM） |
| `basicsr/models/archs/DC_LIDM.py` | 独立 DC-LIDM 模块（plug-and-play） |
| `basicsr/models/losses/losses.py` | Charbonnier / FaceLab / perceptual / gradient / TV 损失 |
| `basicsr/metrics/psnr_ssim.py` | PSNR / SSIM / 掩码内 ΔE |
| `basicsr/my_train.py` · `basicsr/test.py` | 训练 · 评测 |
| `inference.py` | 推理，支持 `--alpha` 闪光控制 |
| `configs/my_aiflash.yaml` · `configs/inference.yaml` | 训练配置 · 推理模板 |

不包含：论文 Table I 的外部对比方法（RetinexNet、SCI、KinD、Zero-DCE、RetinexFormer、NAFNet、
Uformer、plain Restormer），请从其官方仓库自行适配。

## 环境

Python 3.13：

```bash
python -m venv venv
venv\Scripts\activate                                # Windows
pip install -r requirements.txt
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
```

所有命令都在仓库根目录执行。

## 数据

NTIRE 2026 AI Flash Portrait（RAIM Challenge Track 3）：
<https://www.codabench.org/competitions/12885/> —— 数据公开，本仓库**不重分发**。

```
datasets/train_550/{gt,input,mask_personmask}/   # 550 组训练
datasets/val_50/{gt,input,mask_personmask}/      # 50 组验证（论文所有指标）
```

掩码按文件名与图像配对（`input/001.jpg` ↔ `mask_personmask/001.jpg`），并按 `mask > 0` 二值化。

## 权重

从 **Releases** 下载 `best_Restormer_LIDM.pth`，放到 `best_model/best_Restormer_LIDM.pth`。
该权重可复现上面的指标。

## 用法

```bash
# 训练（configs/my_aiflash.yaml：550 组数据、300k 迭代）
python -m basicsr.my_train -opt configs/my_aiflash.yaml

# 评测：先把 path.pretrain_network_g 指向发布权重
# （配置里只保留 val 数据集，即可只评测验证集）
python -m basicsr.test -opt configs/my_aiflash.yaml

# 推理（默认：configs/my_aiflash.yaml + best_model/best_Restormer_LIDM.pth）
python inference.py --input_dir datasets/val_50/input \
                    --mask_dir  datasets/val_50/mask_personmask \
                    --output_dir result/val_50

# 可控闪光强度 α（0 ≈ 重建的低光输入，1 = 训练设定）
python inference.py --alpha 0.5 --output_dir result/alpha0.5

# 显存不足时按块推理（tile 必须为 8 的倍数）；--device cpu 也可用
python inference.py --tile 512
```

推理参数：`--config`、`--weights`、`--input_dir`、`--mask_dir`、`--output_dir`、`--alpha`、
`--tile` / `--tile_overlap`、`--device`（`auto` / `cuda` / `cuda:0` / `cpu`）。

## 说明

- 训练请用 `basicsr/my_train.py`：`basicsr/train.py` 不向模型传人像掩码。
- 指标在网络输出上结合人像掩码计算（`use_image: false`），这就是论文 ΔE 的口径；验证固定 `α = 1`。
- `L_env` 实现为低分辨率空间图，并用 TV 项约束其低频性，因此训练目标为六项（论文 Eq. 7 列了五项）。
- `python basicsr/my_train.py ...` 需先 `pip install -e .`；`python -m basicsr.my_train ...` 可直接用。

## 引用

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

## 许可

Apache License 2.0，见 `LICENSE`。`basicsr/` 框架派生自
[BasicSR](https://github.com/XPixelGroup/BasicSR)，详见 `THIRD_PARTY_NOTICES.md`。