# DC-LIDM — Latent Illumination Decomposition with Dual Reconstruction Constraints for Night-Time Flash Portrait Enhancement

官方代码仓库，对应论文：

> Liqing Wang, Ming Ronnier Luo, *"Latent Illumination Decomposition with Dual Reconstruction Constraints for
> Night-Time Flash Portrait Enhancement"*, IEEE Access (投稿中).

本仓库发布**本文方法（Restormer + DC-LIDM 主线）的实现与权重**。
论文中作为对比的其它方法（RetinexNet / SCI / KinD / Zero-DCE / RetinexFormer / NAFNet / Uformer / plain Restormer）
不在本仓库内，请从各自的官方仓库获取，并按下文 §7 的统一协议适配训练。

---

## 0. 仓库范围（先读这一段）

| 论文内容 | 本仓库是否可复现 | 说明 |
|---|---|---|
| Table I 「Ours」(Restormer+DC-LIDM) | ✅ | `configs/my_aiflash.yaml` + `best_model/best_Restormer_LIDM.pth` |
| Table II 「Full」档 | ✅ | 同上（`Full` = 主模型配置） |
| Table II 「Base.」及 V1/V2/V3 档 | ❌ | 对应 plain Restormer 与历史中间版本，未随仓库分发 |
| Table III（NAFNet / Uformer + DC-LIDM） | ❌ | 需从 NAFNet / Uformer 官方仓库适配；`DC_LIDM.py` 提供了模块接口 |
| Table IV 的对比列 | ❌ | 属其它方法的参数量/FLOPs |
| Fig. 1/2（结构、Retinex 分解） | ✅ | 见 `basicsr/models/archs/my_restormer_arch.py` |
| Fig. 9（R / L / 差值可视化） | ⚠️ | 训练/验证会保存中间量，仓库未附独立可视化脚本 |
| Fig. 12（α 可控闪光） | ✅ | `python inference.py --alpha <v>` |

> 统一对比协议（论文 §IV-A）：同一 550/50 划分、同一输入管线、同一监督损失、ΔE 在**人像掩码内**按 CIE76 平均。
> ΔE 的实现见 `basicsr/metrics/psnr_ssim.py` 的 `calculate_deltaE(img1, img2, mask)`；掩码按 `mask > 0` 二值化。

---

## 1. 目录结构

```
AI-Flash-project/
├── basicsr/                          # 精简后的 BasicSR 框架
│   ├── data/                         # Dataset_MaskedImage / Dataset_PairedImage（含掩码读取）
│   ├── models/
│   │   ├── archs/
│   │   │   ├── my_restormer_arch.py   # ★ 主模型：Restormer_AIFlash_mask_attention + EnvLightHead + FlashHeadMasked
│   │   │   └── DC_LIDM.py             # ★ 独立 DC-LIDM 模块（plug-and-play 接口，供其它 backbone 适配）
│   │   ├── losses/losses.py           # Charbonnier / Charbonnier_mask / FaceLab / Perceptual / Gradient / TVSmooth
│   │   ├── my_ai_flash_model.py       # AIFlashModel：双重建约束、损失组合、best checkpoint
│   │   ├── image_restoration_model.py # 框架自带通用模型
│   │   ├── base_model.py  lr_scheduler.py
│   ├── metrics/psnr_ssim.py           # PSNR / SSIM / ΔE（掩码内 CIE76）
│   ├── utils/                         # logger、options、img_util、file_client ...
│   ├── my_train.py                    # ★ 训练入口（带 mask；不要用 train.py）
│   ├── test.py                        # 验证/测试入口
│   └── visualize.py                   # 中间张量（R/E/F）与 loss 曲线可视化
├── configs/
│   ├── my_aiflash.yaml                # ★ 主模型训练配置（论文 Table I「Ours」/ Table II「Full」）
│   └── inference.yaml                 # 推理/效率分析模板（含 FLOPs、计时、α 可视化开关）
├── best_model/                        # 权重目录（.gitignore 忽略；从 Releases 下载）
├── inference.py                       # ★ 单图/批量推理，支持 --alpha 可控闪光
├── requirements.txt  setup.py  setup.cfg  VERSION
├── readme.txt                         # 早期速记（内容以本文件为准）
└── README.md
```

---

## 2. 环境配置

```bash
# Python 3.13
python -m venv venv
venv\Scripts\activate                 # Windows
pip install -r requirements.txt

# CUDA 版 PyTorch（本机为 RTX 4060Ti 8GB，cu126 示例）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

- `requirements.txt` 已包含训练需要的 `tensorboard`（配置里 `use_tb_logger: true`）与推理需要的 `opencv_python`、`scikit-image`、`natsort`。
- 随 NAFNet / Uformer 一起移除后，**不再需要 `timm` 与 `torch-dwconv`**，环境明显更简单。
- `dlib` 仅在 `basicsr/utils/face_util.py` 中使用，当前流程不依赖（可选）。
- 纯 CPU 也可跑推理（`--device cpu`），只是慢。

---

## 3. 数据准备

数据集来自 **NTIRE 2026 AI Flash Portrait**（Track 3 of the 3rd Restore Any Image Model Challenge），
公开下载入口：<https://www.codabench.org/competitions/12885/>。
本仓库**不分发数据**（请遵守挑战赛数据条款）。

论文使用其中 600 张公开配对中的 **550 训练 / 50 验证**。期望目录结构：

```
datasets/
├── train_550/{gt,input,mask_personmask}/     # 550 组训练
└── val_50/{gt,input,mask_personmask}/        # 50 组验证（论文所有指标都在此划分上报告）

dataset/test/{input,mask_personmask}/         # inference.py 默认输入（跑分/展示）
result/test/                                  # inference.py 默认输出
```

- 每个样本包含：低光输入、闪光 GT、人像掩码。
- 掩码按**文件名**配对（`input/001.jpg` ↔ `mask_personmask/001.jpg`）；缺失时回退为按顺序配对并告警，仍缺失则用全零掩码。
- 掩码读取与训练一致：灰度化后 `mask > 0` 二值化，再由模型内部 `soft_mask = 0.9·mask + 0.1` 使用。
- 训练侧自动把图像裁到 8 的倍数（Restormer 需多次下采样），并按 `gt_size: 256` 随机裁剪 + 几何增强。

---

## 4. 权重

本仓库**只发布主模型权重**：

| 文件 | 架构 | 对应论文 | 指标（val_50） |
|---|---|---|---|
| `best_Restormer_LIDM.pth` | `Restormer_AIFlash_mask_attention` | Table I「Ours」/ Table II「Full」 | PSNR 31.35 / ΔE 2.77 / SSIM 0.968 / 1.54 M / 8.32 G |

- 从 **Releases** 下载后放入 `best_model/best_Restormer_LIDM.pth`（目录被 `.gitignore` 忽略，需手动创建）。
- 权重格式为 `torch.save({'params': state_dict})`，`inference.py` 会自动识别 `params` / `params_ema` / `state_dict` / 单键嵌套存档并剥离 `module.` 前缀。
- 建议校验完整性：`Get-FileHash best_model/best_Restormer_LIDM.pth -Algorithm SHA256`（`sha256sum` on Linux）。
- `best_model/` 下其它本地权重（其它 backbone、消融档）**不参与发布**，不会进入 git。

---

## 5. 训练

```bash
# 必须在仓库根目录执行（basicsr 依赖当前工作目录作为包根）
python basicsr/my_train.py -opt configs/my_aiflash.yaml
```

论文 §IV-B 的实现细节与配置字段对应：

| 论文 | 配置字段 |
|---|---|
| AdamW, β=(0.9,0.999), wd=1e-4, lr 3e-4 → 1e-5（cosine） | `train.optim_g`、`train.scheduler` |
| batch size 1、patch 256×256、单卡 4060Ti 8GB | `datasets.train.batch_size_per_gpu`、`gt_size` |
| α 训练时固定为 1 | `train.alpha_sampling: false` → `alpha = 1.0` |
| Eq.(7) λ₁=2.0 `L_rec` | `train.pixel_opt`（`CharbonnierLoss_mask`, 2） |
| Eq.(7) λ₂=1.0 `L_input` | `train.input_recon_opt`（`CharbonnierLoss`, 1.0） |
| Eq.(7) λ₃=1.0 `L_Lab`（λ_L=1.5, λ_ab=1.0） | `train.luminance_opt`（`FaceLabLoss`） |
| Eq.(7) λ₄=0.001 `L_percept`（conv3_4 / conv4_4） | `train.perceptual_opt` |
| Eq.(7) λ₅=0.2 `L_grad` | `train.grad_opt` |
| **未被论文 Eq.(7) 列出** | `train.smooth_opt`（`TVSmoothLoss`, 0.1）——见 §7 不一致说明 |

输出与产物：

- 权重与状态：`experiments/<name>/models/`（`best_net_g_psnr.pth`、`best_net_g_deltaE.pth`、`net_g_latest.pth`、`best_metrics.json`）
- 日志：`experiments/<name>/train_*.log`（开头即完整配置打印，便于回溯）
- 可视化：`experiments/<name>/val_visualizations/`（R / L_env / L_flash 中间量）、loss 曲线
- 断点续训：自动读取 `experiments/<name>/training_states/` 下最大的 `.state`

> 注意：`basicsr/train.py` 不向模型传 `mask`，配 `model_type: AIFlashModel` 会 `KeyError: 'mask'`；**请统一使用 `my_train.py`**。

---

## 6. 推理（含可控闪光 α）

```bash
# 默认：configs/my_aiflash.yaml + best_model/best_Restormer_LIDM.pth
#       输入 dataset/test/{input,mask_personmask} → 输出 result/test/
python inference.py

# 指定配置 / 权重 / 目录
python inference.py --config configs/my_aiflash.yaml \
                    --weights best_model/best_Restormer_LIDM.pth \
                    --input_dir dataset/test/input \
                    --mask_dir dataset/test/mask_personmask \
                    --output_dir result/test

# 论文 Eq.(1) 的 α：α=0 → 重建的低光图 R×L_env；α=1 → 训练设定；>1 → 更强闪光（Fig. 12）
python inference.py --alpha 0     --output_dir result/alpha0.0
python inference.py --alpha 0.5   --output_dir result/alpha0.5
python inference.py --alpha 1.5   --output_dir result/alpha1.5
python inference.py --alpha 2.0   --output_dir result/alpha2.0

# 显存不足时按块推理（tile 需为 8 的倍数）
python inference.py --tile 512 --tile_overlap 32

# 单张图片 / 强制 CPU / 指定 GPU
python inference.py --input_dir dataset/test/input/001.jpg \
                    --mask_dir dataset/test/mask_personmask/001.jpg
python inference.py --device cpu
python inference.py --device cuda:0
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `configs/my_aiflash.yaml` | 仅使用其中的 `network_g` 段构建网络 |
| `--weights` | `best_model/best_Restormer_LIDM.pth` | 权重路径 |
| `--input_dir` | `dataset/test/input` | 目录或单张图片 |
| `--mask_dir` | `dataset/test/mask_personmask` | 目录或单张掩码 |
| `--output_dir` | `result/test` | 输出目录（自动创建，输出 `.png`） |
| `--alpha` | `1.0` | 闪光强度 α（论文 Eq.(1)） |
| `--tile` / `--tile_overlap` | `None` / `32` | 分块推理；不设置则整图推理 |
| `--device` | `auto` | `auto` / `cuda` / `cuda:0` / `cpu`，cuda 不可用时自动回退 |

行为说明：图像先 pad 到 8 的倍数再推理，输出裁回原尺寸并 clamp 到 `[0,1]`；逐张打印进度与失败原因，全部失败时以退出码 `1` 结束。

---

## 7. 论文 ↔ 代码 对照

| 论文 | 代码位置 |
|---|---|
| Eq.(1) `I = R × (L_env + α·L_flash)` | `my_restormer_arch.py` → `Restormer_AIFlash_mask_attention.forward`：`illumination = env_light_img + alpha * flash_map` |
| Eq.(2) `L_env` 全局估计 | `EnvLightHead` + forward 中的 `0.9·σ(·)` 与上采样（**与论文描述不一致，见下**） |
| Eq.(3) 掩码作 query 的交叉注意力 | `FlashHeadMasked.forward`：mask → Q，latent → K/V，含 `torch.sigmoid` 前的缩放 |
| Eq.(4) 门控融合 `(1-g)⊙F + g⊙F̂` | `FlashHeadMasked` 的 `gate_conv`（两层卷积 + sigmoid 生成门） |
| Eq.(5) `L_flash = σ(D(F')) ⊙ (γM + 1 − γ)`, γ=0.9 | `soft_mask = 0.9 * mask + 0.1`，`flash_map = flash_map * soft_mask` |
| Eq.(6) 双重建约束 | `my_ai_flash_model.py` → `optimize_parameters`：`input_recon = R × L_env` 对齐 `srgb_to_linear(lq)`；GT 侧由 `pixel_opt` 约束 `R × (L_env + α·L_flash) ≈ I_gt` |
| sRGB ↔ 线性 RGB | `srgb_to_linear` / `linear_to_srgb`（`my_restormer_arch.py` 末尾） |
| Eq.(7) 损失加权 | `configs/my_aiflash.yaml` 的 `train.*_opt.loss_weight` |
| ΔE（掩码内 CIE76） | `basicsr/metrics/psnr_ssim.py` → `calculate_deltaE(img1, img2, mask)` |
| 损失实现 | `basicsr/models/losses/losses.py`（`FaceLabLoss` / `CharbonnierLoss_mask` / `PerceptualLoss` / `GradientLoss` / `TVSmoothLoss`） |

### ⚠️ 发布前建议先修正的三处论文—代码不一致

1. **`L_env` 是空间图，不是全局 1×1×3。** 论文摘要、Fig. 1 caption 与 Eq.(2)（GAP + 逐通道回归）都描述为
   全局环境光；代码中 `EnvLightHead` 输出 `(B,3,H/16,W/16)` 的空间分布图（高斯平滑），再 `0.9·σ(·)` 后**双线性上采样**到全分辨率，
   并由额外的 `TVSmoothLoss` 约束其低频性。建议**以代码为准修改论文**（Eq.(2)、§III-B 文字、Fig. 1 caption），
   否则读者按论文描述会无法对上代码；若要保留"全局"表述，则需重训为全局版本。
2. **损失项数：论文 Eq.(7) 列 5 项，代码/config 有 6 项**（多出 `TVSmoothLoss`, 权重 0.1）。建议把 TV 项作为 λ₆ 写入 Eq.(7)。
3. **训练迭代数：论文写 200,000 iterations，而发布配置是 300,000**（`total_iter`/`iters` 与 cosine 周期和一致），
   且发布的 `best_Restormer_LIDM.pth` 内部记录名为 `net_g_latest`（即某次训练的最后一次保存）。
   请核对发布权重实际的迭代数/选择依据，并与论文表述统一。

---

## 8. 许可与引用

- ⚠️ **仓库目前没有 LICENSE。** `basicsr/` 派生自 [BasicSR](https://github.com/XPixelGroup/BasicSR)（Apache-2.0, © Xintao Wang），
  Apache-2.0 要求保留版权与许可声明；主模型结构参考了 Restormer。发布前请补：
  - `LICENSE`（建议 Apache-2.0）
  - `THIRD_PARTY_NOTICES.md`（写明 BasicSR 的来源与许可、Restormer 架构参考、NTIRE 数据集条款）
  否则默认"保留所有权利"，与论文里 "source code ... publicly available" 的表述不匹配。
- 引用（BibTeX，DOI 待分配）：

```bibtex
@article{wang2026dclidm,
  title   = {Latent Illumination Decomposition with Dual Reconstruction Constraints
             for Night-Time Flash Portrait Enhancement},
  author  = {Wang, Liqing and Luo, Ming Ronnier},
  journal = {IEEE Access},
  year    = {2026},
  note    = {to appear}
}
```

---

## 9. 已知问题与验证状态

已知问题（均不影响主模型训练/推理主流程）：

1. `basicsr/__init__.py` 缺失 → `setup.py` 的 `find_packages()` 收集不到包，`pip install -e .` 无效；请**在仓库根目录运行**脚本。
2. `basicsr/train.py` 不传 mask（请用 `my_train.py`）。
3. `basicsr/metrics/fid.py` 引用了不存在的 `basicsr.models.archs.inception`（当前流程不使用 fid）。
4. `Dataset_GaussianDenoising` 中 `from basicsr.utils.scandir import scandir` 路径有误（应为 `basicsr.utils`；当前流程不使用该类）。

验证状态：

- ✅ 全部 Python 文件通过语法编译；包内导入解析仅剩上述两处未使用路径的问题。
- ✅ 已删除论文未使用的 RetinexMamba / SwinIR（仅出现在 Related Work），`import basicsr.models.archs` 不再中断。
- ✅ `inference.py` 的 CLI 参数、路径解析、100 张样例图与掩码按名配对、tile 分支调用、α 透传、缺文件报错路径，均通过桩测试（stub torch 静态验证）。
- ⚠️ **未做真实权重的前向/指标实跑**：开发环境未安装 `torch`。首次使用建议先跑
  `python inference.py --input_dir <单张图> --device cpu`，再跑全量验证。