# DC-LIDM：用于夜间闪光人像增强的潜在光照分解与双重建约束

[English](README.md) | **中文**

本仓库是以下论文的官方代码发布：

> Liqing Wang, Ming Ronnier Luo, *"Latent Illumination Decomposition with Dual Reconstruction
> Constraints for Night-Time Flash Portrait Enhancement"*, IEEE Access（审稿中）。

DC-LIDM 把低光人像分解为**反射率 `R`**、**全局环境光 `L_env`** 与**空间变化的闪光图 `L_flash`**，
在**线性 RGB** 空间按 `I = R × (L_env + α·L_flash)` 重组，并用**掩码引导的交叉注意力**
（人像掩码作为 query）预测闪光图。两侧重建约束（输入侧与 GT 侧）共享同一张反射率，从两端正则化分解。
推理时用单个标量 `α` 连续控制闪光强度。

**主模型**：Restormer backbone + DC-LIDM，1.54 M 参数，256×256 输入 8.32 G FLOPs，
在 50 张验证集上 **PSNR 31.35 / ΔE 2.77 / SSIM 0.968**（论文 Table I 的 "Ours" 行）。

---

## 1. 仓库范围（包含 / 不包含什么）

| 内容 | 是否包含 | 说明 |
|---|---|---|
| 主模型 `Restormer_AIFlash_mask_attention` | ✅ | `basicsr/models/archs/my_restormer_arch.py` |
| 独立 DC-LIDM 模块（`DC_LIDM.py`，plug-and-play 接口） | ✅ | 供插入其它 encoder–decoder backbone |
| 损失 / 指标（Charbonnier、FaceLab、perceptual、gradient、TV；PSNR/SSIM/ΔE） | ✅ | `basicsr/models/losses/`、`basicsr/metrics/` |
| 训练 / 评测 / 推理入口 | ✅ | `basicsr/my_train.py`、`basicsr/test.py`、`inference.py` |
| 主模型预训练权重 | ✅（GitHub Releases） | `best_Restormer_LIDM.pth`，见 §5 |
| 完整训练数据（550 组） | ❌ | 公开赛事数据，请从赛事官网获取（§4） |
| Table I 的 7 个外部对比方法 | ❌ | RetinexNet / SCI / KinD / Zero-DCE / RetinexFormer / NAFNet / Uformer / plain Restormer，请从其官方仓库自行适配 |
| 跨 backbone 版本（NAFNet+DC-LIDM、Uformer+DC-LIDM） | ❌ | 请把 `DC_LIDM.py` 按其官方代码接入对应 backbone |
| 消融中间版本 V1/V2/V3 | ❌ | 历史中间模型，未发布 |

论文采用的**统一对比协议**（本仓库代码即可复现该协议）：同一 550/50 划分、同一输入管线、
同一监督损失、`crop_border=0`、不做 Y 通道转换、**ΔE 在人像掩码内**按 CIELAB 的 CIE76 平均。

---

## 2. 已实测跑通的完整链路

硬件 / 软件：**NVIDIA RTX 4060 Laptop GPU (8 GB)，CUDA 12.6，
torch 2.10.0+cu126，torchvision 0.25.0+cu126，Python 3.13.11**（见 `requirements.txt`）。

| 环节 | 命令 | 实测结果 |
|---|---|---|
| **训练**（冒烟：1 张样本、20 次迭代） | `python -m basicsr.my_train -opt configs/smoke_val1.yaml` | 退出码 0，23 s；6 项损失全部参与；iter 10 / 20 各做一次验证；产出 `best_net_g_psnr.pth`、`best_net_g_deltaE.pth`、`net_g_latest.pth`、`.state`、loss 曲线、TensorBoard 事件与 R/E/F/光照可视化。再次运行可**正确自动续训** |
| **评测**（发布权重 + 50 张验证图） | `python -m basicsr.test -opt configs/eval_val50.yaml` | 退出码 0，48 s → **PSNR 31.3518 / ΔE 2.7683 / SSIM 0.9676**，复现论文 Table I "Ours"（31.35 / 2.77 / 0.968） |
| **推理**（批量） | `python inference.py` | 10/10 张写入 `result/test/`，退出码 0 |
| **推理**（发布权重） | `python inference.py --config configs/my_aiflash.yaml --weights best_model/best_Restormer_LIDM.pth` | `α=1` 与 `α=0` 均 10/10，退出码 0 |
| **分块推理** | `python inference.py --tile 512 --tile_overlap 32` | 与整图推理平均差 **1.62/255**（最大 21/255，仅接缝融合差异） |
| **闪光强度 `α`** | `python inference.py --alpha 0 / 0.5 / 1 / 2` | 平均亮度单调上升 0.527 → 0.568 → 0.602 → 0.660（用本地冒烟权重量测） |
| **输入侧约束（Eq. 6）** | 发布权重，`--alpha 0` | `mean|α=0 − input| = 3.36/255`（1.3%），说明 `R×L_env` 确实重建了低光输入 |
| **CPU 推理** | `python inference.py --device cpu` | 单张 1024×768 约 10.8 s |

> 冒烟模型只训练了 20 次迭代，它的 PSNR/ΔE 数值没有质量意义，仅用于证明"训练→验证→评测→推理"
> 整条链路可跑通。

---

## 3. 目录结构

```
AI-Flash-project/
├── basicsr/                              # 精简后的 BasicSR 框架
│   ├── __init__.py
│   ├── data/                             # Dataset_MaskedImage / Dataset_PairedImage（含掩码读取）
│   ├── models/
│   │   ├── archs/
│   │   │   ├── my_restormer_arch.py       # 主模型 + EnvLightHead + FlashHeadMasked + sRGB<->linear
│   │   │   └── DC_LIDM.py                 # 独立 DC-LIDM 模块（plug-and-play）
│   │   ├── losses/losses.py               # Charbonnier / Charbonnier_mask / FaceLab / Perceptual / Gradient / TVSmooth
│   │   ├── my_ai_flash_model.py           # AIFlashModel：双约束、损失组装、best checkpoint
│   │   ├── image_restoration_model.py     # 框架通用模型
│   │   ├── base_model.py  lr_scheduler.py
│   ├── metrics/psnr_ssim.py               # PSNR / SSIM / ΔE（掩码内 CIE76）
│   ├── utils/
│   ├── my_train.py                        # 训练入口（带 mask）—— 用这个
│   ├── test.py                            # 评测入口
│   ├── train.py                           # 原始入口（不传 mask）
│   └── visualize.py                       # 中间张量与 loss 曲线可视化
├── configs/
│   ├── my_aiflash.yaml                    # 论文训练配置（Table I "Ours" / Table II "Full"）
│   ├── eval_val50.yaml                    # 在 val_50 上复现 Table I "Ours"
│   ├── smoke_val1.yaml                    # 1 样本 / 20 迭代冒烟测试
│   └── inference.yaml                     # 推理 / 效率分析模板
├── best_model/                            # 放置发布权重（已被 .gitignore 忽略）
├── inference.py                           # 单图与批量推理，支持 --alpha
├── requirements.txt  setup.py  setup.cfg  VERSION
├── LICENSE  THIRD_PARTY_NOTICES.md
└── README.md  README_zh-CN.md  readme.txt
```

---

## 4. 数据准备

数据集：**NTIRE 2026 AI Flash Portrait**（第 3 届 Restore Any Image Model Challenge 的 Track 3），
下载入口 <https://www.codabench.org/competitions/12885/>。
本仓库**不重分发数据**，请遵守赛事数据条款。

论文把 600 张公开配对划分为 **550 训练 / 50 验证**：

```
datasets/
├── train_550/{gt,input,mask_personmask}/
└── val_50/{gt,input,mask_personmask}/      # 论文所有指标都在此划分上报告

datasets/val_1/{gt,input,mask_personmask}/  # 1 张样本，冒烟配置使用
dataset/test/{input,mask_personmask}/       # inference.py 默认输入
result/test/                                # inference.py 默认输出
```

掩码按**文件名**与图像配对（`input/001.jpg` ↔ `mask_personmask/001.jpg`），并按 `mask > 0`
二值化（与训练一致）。图像会被取整/裁剪到 8 的倍数，以满足多尺度 backbone 的下采样要求。

---

## 5. 权重

只发布主模型权重，作为 GitHub **Release** 附件：

| 文件 | 架构 | 可复现的结果 |
|---|---|---|
| `best_Restormer_LIDM.pth` | `Restormer_AIFlash_mask_attention` | PSNR 31.35 / ΔE 2.77 / SSIM 0.968（Table I "Ours"） |

1. 从仓库的 **Releases** 页面下载；
2. 放到 `best_model/best_Restormer_LIDM.pth`（该目录已被 gitignore，不存在请新建）；
3. 可选校验：`Get-FileHash best_model/best_Restormer_LIDM.pth -Algorithm SHA256`（Linux 用 `sha256sum`）。

权重为 `torch.save({'params': state_dict})` 格式；加载器同时兼容 `params_ema`、`state_dict`
以及单键嵌套存档，并会自动去掉 `module.` 前缀。

---

## 6. 环境安装

```bash
# Python 3.13
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# CUDA 版 PyTorch（以 CUDA 12.6 为例）
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
```

所有命令都在**仓库根目录**执行：

```bash
pip install -e .                                            # 可选，装后可用 `python basicsr/my_train.py ...`
python -m basicsr.my_train -opt configs/my_aiflash.yaml      # 始终可用
```

`python -m ...` 会把仓库根目录加入导入路径，因此即使环境里存在指向别处的 `basicsr`
可编辑安装，`import basicsr` 也会解析到本仓库的代码。

---

## 7. 训练

```bash
# 论文完整训练（300k 迭代、550 组数据）—— configs/my_aiflash.yaml
python -m basicsr.my_train -opt configs/my_aiflash.yaml

# 单样本冒烟测试（20 次迭代，约 25 s）—— 验证整条链路
python -m basicsr.my_train -opt configs/smoke_val1.yaml
```

配置项与论文的对应关系：

| 论文 | 配置字段 |
|---|---|
| AdamW，β=(0.9, 0.999)，wd 1e-4，lr 3e-4 余弦退火 | `train.optim_g`、`train.scheduler` |
| batch size 1，256×256 patch，单张 8 GB 显卡 | `datasets.train.batch_size_per_gpu`、`gt_size` |
| 训练时 α = 1 | `train.alpha_sampling: false` |
| λ₁ `L_rec` = 2.0 | `train.pixel_opt`（`CharbonnierLoss_mask`） |
| λ₂ `L_input` = 1.0 | `train.input_recon_opt`（`CharbonnierLoss`） |
| λ₃ `L_Lab` = 1.0（λ_L 1.5，λ_ab 1.0） | `train.luminance_opt`（`FaceLabLoss`） |
| λ₄ `L_percept` = 0.001（conv3_4、conv4_4） | `train.perceptual_opt` |
| λ₅ `L_grad` = 0.2 | `train.grad_opt` |
| 额外的 `L_env` TV 平滑项（论文 Eq. 7 未列出） | `train.smooth_opt`（`TVSmoothLoss`, 0.1） |

产出：`experiments/<name>/` 下的 `models/`（`best_net_g_psnr.pth`、`best_net_g_deltaE.pth`、
`net_g_latest.pth`、`best_metrics.json`）、`training_states/`（`.state` + loss 历史）、
`loss_curves/`、`visualization/`、`val_visualizations/` 与完整训练日志；
TensorBoard 日志在 `tb_logger/<name>/`。若 `training_states/` 下已有 `.state`，
训练会自动从最大的那个续训。

---

## 8. 评测（复现 Table I 的 "Ours"）

```bash
# 需要 best_model/best_Restormer_LIDM.pth 与 datasets/val_50/
python -m basicsr.test -opt configs/eval_val50.yaml
```

预期日志（实测 PSNR 31.3518 / ΔE 2.7683 / SSIM 0.9676，48 s）：

```
INFO: Validation ValSet50,   # psnr: 31.3518   # deltaE: 2.7683   # ssim: 0.9676
```

---

## 9. 推理

```bash
# 默认：configs/my_aiflash.yaml + best_model/best_Restormer_LIDM.pth
#      输入 dataset/test/{input,mask_personmask} → 输出 result/test/
python inference.py

python inference.py --config configs/my_aiflash.yaml \
                    --weights best_model/best_Restormer_LIDM.pth \
                    --input_dir dataset/test/input \
                    --mask_dir dataset/test/mask_personmask \
                    --output_dir result/test

# 可控闪光强度 α（论文 Eq. 1 / Fig. 12）
python inference.py --alpha 0     --output_dir result/alpha0.0   # ≈ 重建的低光输入
python inference.py --alpha 0.5   --output_dir result/alpha0.5
python inference.py --alpha 1.5   --output_dir result/alpha1.5
python inference.py --alpha 2.0   --output_dir result/alpha2.0

# 显存不足时按块推理（tile 必须为 8 的倍数）
python inference.py --tile 512 --tile_overlap 32

# 单张图片 / CPU / 指定 GPU
python inference.py --input_dir dataset/test/input/001.jpg \
                    --mask_dir dataset/test/mask_personmask/001.jpg
python inference.py --device cpu
python inference.py --device cuda:0
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `configs/my_aiflash.yaml` | 只用其中的 `network_g` 段构建网络 |
| `--weights` | `best_model/best_Restormer_LIDM.pth` | 权重路径 |
| `--input_dir` | `dataset/test/input` | 目录或单张图片 |
| `--mask_dir` | `dataset/test/mask_personmask` | 目录或单张掩码 |
| `--output_dir` | `result/test` | 自动创建，输出 PNG |
| `--alpha` | `1.0` | 闪光强度，对应论文 Eq. (1) |
| `--tile` / `--tile_overlap` | `None` / `32` | 分块推理，未设置则整图推理 |
| `--device` | `auto` | `auto` / `cuda` / `cuda:0` / `cpu`，cuda 不可用时自动回退 |

图像先 pad 到 8 的倍数再推理，输出裁回原尺寸并 clamp 到 `[0,1]`；逐张打印进度，全部失败时退出码为 1。

---

## 10. 论文 ↔ 代码 对照

| 论文 | 代码 |
|---|---|
| Eq. (1) `I = R × (L_env + α·L_flash)` | `Restormer_AIFlash_mask_attention.forward`：`illumination = env_light_img + alpha * flash_map` |
| Eq. (2) 环境光估计 | `EnvLightHead` + `0.9·σ(·)` + 双线性上采样（见"实现说明"） |
| Eq. (3) 掩码作 query 的交叉注意力 | `FlashHeadMasked.forward`（mask → Q，latent → K/V） |
| Eq. (4) 门控融合 `(1-g)⊙F + g⊙F̂` | `FlashHeadMasked` 的 `gate_conv`（两层卷积 + sigmoid 门） |
| Eq. (5) `L_flash = σ(D(F')) ⊙ (γM + 1 − γ)`，γ = 0.9 | `soft_mask = 0.9 * mask + 0.1` |
| Eq. (6) 双重建约束 | `my_ai_flash_model.optimize_parameters`：`input_recon = R × L_env` 对齐 `srgb_to_linear(lq)`；GT 侧由 `pixel_opt` 约束 |
| Eq. (7) 损失加权 | `configs/my_aiflash.yaml` → `train.*_opt.loss_weight` |
| 线性 RGB 处理 | `srgb_to_linear` / `linear_to_srgb` |
| ΔE（掩码内 CIE76） | `basicsr/metrics/psnr_ssim.py: calculate_deltaE(img1, img2, mask)` |
| 损失实现 | `basicsr/models/losses/losses.py` |

**实现说明**

- `L_env` 在代码中实现为**低分辨率空间图** `(B, 3, H/16, W/16)`：卷积头 + 高斯平滑，乘 `0.9·σ(·)`
  后双线性上采样到输出分辨率，并用 `TVSmoothLoss`（`train.smooth_opt`）约束其低频性。
  因此训练目标实际包含**六项**，而论文 Eq. (7) 列出五项。
- 验证/评测固定使用 `α = 1`。
- 所有配置都使用 `use_image: false`，即指标直接在 tensor 上结合人像掩码计算（这就是论文 ΔE 的口径）；
  `use_image: true` 分支与"带 mask 的指标签名"不匹配，未支持。

---

## 11. 已知问题

1. `basicsr/train.py` 不向模型传人像掩码，因此无法训练 `AIFlashModel`（会 `KeyError: 'mask'`）。
   请使用 `basicsr/my_train.py`。
2. `pad_test()` 只对低光输入做 pad，没有同步 pad 掩码；仅当输入高/宽不是 8 的倍数时才有影响
   （数据集已把尺寸取整到 8 的倍数）。
3. `basicsr/metrics/fid.py` 引用了本仓库不存在的模块（FID 未被使用）。
4. `Dataset_GaussianDenoising` 里 `from basicsr.utils.scandir import scandir` 路径有误（该类未被使用）。
5. `python basicsr/my_train.py ...` 需先 `pip install -e .` 才能用；
   否则请用 `python -m basicsr.my_train ...`。
6. 外部对比方法与跨 backbone 版本按设计未包含（见 §1），
   因此 Table I 的其它行、Table III 与 Table IV 的对比列无法仅凭本仓库复现。

---

## 12. 许可与引用

本项目以 **Apache License 2.0** 发布（见 `LICENSE`）。`basicsr/` 目录派生自
[BasicSR](https://github.com/XPixelGroup/BasicSR)（Apache-2.0，© Xintao Wang）；
主模型遵循 Restormer 架构。详见 `THIRD_PARTY_NOTICES.md`。

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

若使用 NTIRE 2026 AI Flash Portrait 数据集，请同时引用赛事报告（论文参考文献中的 `guan2026`）
并遵守其数据条款。