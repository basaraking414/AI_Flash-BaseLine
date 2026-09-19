# AI-Flash: 人像闪光增强（Illumination Decomposition + Mask Attention）

基于 BasicSR 框架的人像「闪光/补光」增强项目：把低光人像分解为**反射率 R** 与**光照 E**，
再叠加由人像掩码引导的**闪光分量 F**（`I = R × (E + α·F)`），并用多种 backbone 做横向对比。

- **训练/验证框架**：`basicsr/`（自带精简版 BasicSR，含数据、模型、指标、工具）
- **可训练配置**：`configs/*.yaml`
- **推理脚本**：`inference.py`
- **最佳权重**：`best_model/`（**不入库**，见下文「权重与数据」）

> ⚠️ 本仓库默认**不包含**权重与数据集（体积原因），首次使用请按「权重与数据」一节放置文件。
> 另有一处会阻断运行的已知问题，见文末「已知问题」。

---

## 1. 目录结构

```
AI-Flash-project/
├── basicsr/                     # 框架与训练入口
│   ├── data/                    # Dataset_MaskedImage / Dataset_PairedImage（含掩码读取）
│   ├── models/
│   │   ├── archs/               # 各 backbone 与 AI-Flash/LIDM 变体
│   │   ├── losses/              # Charbonnier / FaceLab / TVSmooth / Gradient / Perceptual ...
│   │   ├── my_ai_flash_model.py # AIFlashModel（掩码 + 多损失 + best checkpoint）
│   │   └── base_model.py, lr_scheduler.py
│   ├── metrics/                 # PSNR / SSIM / ΔE / NIQE
│   ├── utils/                   # logger、options、img_util ...
│   ├── my_train.py              # ★ 训练入口（带 mask，推荐）
│   ├── train.py                 # 原始训练入口（不带 mask，仅适用于 ImageCleanModel）
│   ├── test.py                  # 验证/测试入口
│   └── visualize.py             # 中间张量（R/E/F）与 loss 曲线可视化
├── configs/                     # 各 backbone 的训练/推理配置
├── best_model/                  # 最佳权重（.gitignore 忽略）
├── inference.py                 # ★ 推理脚本
├── requirements.txt             # 依赖（缺失项见「已知问题」）
├── setup.py / setup.cfg / VERSION
└── readme.txt                   # 原始速记（与 README.md 内容重叠）
```

---

## 2. 环境配置

```bash
# Python 3.13
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# CUDA 版 PyTorch（本机为 GTX 4060 8GB，cu126 示例）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

`requirements.txt` 目前**遗漏**了若干运行期必需的包（原因见「已知问题」），请补装：

```bash
pip install timm tensorboard torch-dwconv
# dlib 仅在 basicsr/utils/face_util.py 中使用，可选
# pip install dlib
```

> 纯 CPU 也能跑 `inference.py`（`--device cpu`），只是慢。

---

## 3. 权重与数据（不入库，需自行放置）

### 3.1 数据集

训练脚本按 `configs/*.yaml` 中的相对路径读取数据，**必须在仓库根目录运行**。
期望的目录结构（`datasets/` 为训练用，`dataset/` 为推理脚本默认读取位置，注意单复数区别）：

```
datasets/
├── train_550/
│   ├── gt/                 # 正常曝光参考图
│   ├── input/              # 低光输入
│   └── mask_personmask/    # 人像掩码（灰度）
├── val_50/
│   ├── gt/  input/  mask_personmask/
└── test/
    ├── input/
    └── mask_personmask/

dataset/                    # inference.py 默认输入
└── test/
    ├── input/
    └── mask_personmask/

result/test/                # inference.py 默认输出（自动创建）
```

要点：

- 掩码按**文件名**与图像配对（不含扩展名），例如 `input/001.jpg` ↔ `mask_personmask/001.jpg`；
  找不到时回退为按顺序配对，仍找不到则使用全零掩码并在日志中告警。
- 掩码读取方式与训练保持一致：灰度化后按 `mask > 0` 二值化（`{0,1}`），
  再经模型内部 `soft_mask = 0.9·mask + 0.1` 使用。
- 训练侧数据集会自动把图裁到 8 的倍数（Restormer 需多次下采样），并按 `gt_size` 随机裁剪 + 几何增强。

### 3.2 权重

把权重放入 `best_model/`。已核实每个文件的参数前缀与其架构一致
（例如 `best_Restormer_LIDM.pth` 含 `env_head/flash_head`，`best_*_LIDM.pth` 含 `dc_lidm.` 前缀）。

| 权重文件 | 架构类 | 对应配置 | 备注 |
|---|---|---|---|
| `best_Restormer_LIDM.pth` | `Restormer_AIFlash_mask_attention` | `configs/my_aiflash.yaml`、`configs/inference.yaml` | **推理默认权重** |
| `best_NAFNet_baseline.pth` | `NAFNet` | `configs/nafnet.yaml` | |
| `best_NAFNet_LIDM.pth` | `NAFNet_LIDM` | `configs/nafnet_dclidm.yaml` | |
| `best_uformer_baseline.pth` | `Uformer_AIFlash` | `configs/uformer.yaml` | 需 `torch-dwconv` |
| `best_uformer_LIDM.pth` | `Uformer_LIDM` | `configs/uformer_dclidm.yaml` | 需 `torch-dwconv` |
| `best_SCI_baseline.pth` | `SCI_Net` | `configs/sci.yaml` | |
| `best_Restormer_baseline.pth` | `Restormer` | — | 缺配置 |
| `best_KinD_baseline.pth` | `KinDNet` | — | 缺配置 |
| `best_RetinexFormer_baseline.pth` | `RetinexFormer` | — | 缺配置 |
| `best_RetinexNet_baseline.pth` | `RetinexNet` | — | 缺配置 |
| `best_ZeroDCE_baseline.pth` | `ZeroDCE` | — | 缺配置 |

所有权重均为 `torch.save({'params': state_dict}, ...)` 格式，`inference.py` 会自动识别
`params` / `params_ema` / `state_dict` / 单键嵌套存档，并去除 `module.` 前缀。

---

## 4. 训练

```bash
# 在仓库根目录执行
python basicsr/my_train.py -opt configs/my_aiflash.yaml
```

- 使用 **`my_train.py`**（不是 `train.py`）：它会把 `mask` 一起送入模型，
  并按 PSNR / ΔE 分别保存最佳权重到 `experiments/<name>/models/`
  （`best_net_g_psnr.pth`、`best_net_g_deltaE.pth`），同时记录 `best_metrics.json` 与 loss 曲线。
- val 数据集在配置里使用 `Dataset_PairedImage`，训练集使用 `Dataset_MaskedImage`；
  两者都会返回 `mask_personmask`。
- 指标计算走 `use_image: false` 分支（直接用 tensor + mask 计算 PSNR/SSIM/ΔE），
  该分支内部固定使用 `cuda`，纯 CPU 训练/验证需注意。
- 断点续训：自动读取 `experiments/<name>/training_states/` 下最大的 `.state` 文件。

对比实验使用的配置：

| 配置 | 网络 | 说明 |
|---|---|---|
| `configs/my_aiflash.yaml` | Restormer + mask attention | 主线模型 |
| `configs/nafnet.yaml` / `nafnet_dclidm.yaml` | NAFNet / NAFNet+LIDM | |
| `configs/uformer.yaml` / `uformer_dclidm.yaml` | Uformer_AIFlash / Uformer_LIDM | 需 `torch-dwconv` |
| `configs/sci.yaml` | SCI_Net | |
| `configs/retinexmamba*.yaml` | RetinexMamba | **当前不可用**，见「已知问题」 |
| `configs/swinir.yaml` | SwinIR_AIFlash | **当前不可用**，见「已知问题」 |
| `configs/DC_LIDM.yaml` | NAFNet_AIFlash_LIDM | **当前不可用**，见「已知问题」 |

## 5. 验证 / 测试

```bash
python basicsr/test.py -opt configs/my_aiflash.yaml
```

---

## 6. 推理

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

# 显存不足时按块推理（tile 需为 8 的倍数）
python inference.py --tile 512 --tile_overlap 32

# 单张图片 / 强制 CPU / 指定 GPU
python inference.py --input_dir dataset/test/input/001.jpg \
                    --mask_dir dataset/test/mask_personmask/001.jpg
python inference.py --device cpu
python inference.py --device cuda:0
```

参数一览：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `configs/my_aiflash.yaml` | 只使用其中的 `network_g` 段构建网络 |
| `--weights` | `best_model/best_Restormer_LIDM.pth` | 权重路径 |
| `--input_dir` | `dataset/test/input` | 目录或单张图片 |
| `--mask_dir` | `dataset/test/mask_personmask` | 目录或单张掩码 |
| `--output_dir` | `result/test` | 输出目录（自动创建，输出为 `.png`） |
| `--tile` / `--tile_overlap` | `None` / `32` | 分块推理，未设置则整图推理 |
| `--device` | `auto` | `auto` / `cuda` / `cuda:0` / `cpu`，cuda 不可用时自动回退 |

行为说明：图像先 pad 到 8 的倍数再推理，输出裁回原尺寸并 clamp 到 `[0,1]`；
逐张打印进度与失败原因，全部失败时以退出码 `1` 结束。

---

## 7. 已知问题（按影响排序）

1. **【阻断】`basicsr/models/archs/RetinexMamba/` 子包缺失。**
   `RetinexMamba_arch.py` 顶层 `from basicsr.models.archs.RetinexMamba.SS2D_arch import SS2D`，
   而该目录不存在 → `import basicsr.models.archs` 直接抛 `ModuleNotFoundError`。
   由于 `archs/__init__.py` 会自动导入每个 `*_arch.py`，**训练和推理都会在导入阶段中断**。
   两种修法：
   - 恢复 `SS2D_arch.py` / `IFA_arch.py`（上游 RetinexMamba 仓库的 `basicsr/models/archs/` 下）并安装 `mamba-ssm`（仅 Linux 可编译）；
   - 或给这两行 import 加 `try/except` 守卫，使 RetinexMamba 成为可选模块（推荐在 Windows 上使用）。
2. **`requirements.txt` 缺依赖**：`timm`（`Uformer_arch.py`、`SwinIR_arch.py` 顶层导入，缺了同样会让整个 `archs` 导入失败）、
   `tensorboard`（`use_tb_logger: true` 时必需）、`torch-dwconv`（Uformer 的 `FastLeFF`）、`dlib`（可选）。
   另外 `timm.models.layers` 在 timm ≥ 1.0 已废弃，建议锁定 `timm<1.0`（如 `0.9.16`）或改用 `timm.layers`。
3. **两个配置引用了不存在的网络类**：`configs/DC_LIDM.yaml → NAFNet_AIFlash_LIDM`、
   `configs/swinir.yaml → SwinIR_AIFlash`，建模时会 `ValueError: ... is not found`。
4. **权重与配置覆盖不齐**：5 个 baseline 权重（KinD / Restormer / RetinexFormer / RetinexNet / ZeroDCE）缺配置；
   `retinexmamba*.yaml`、`swinir.yaml` 有配置但无权重。
5. **`basicsr/__init__.py` 缺失**：`setup.py` 的 `find_packages()` 收集不到包，`pip install -e .` 无效；
   目前依赖「在仓库根目录运行」的命名空间包机制。
6. **`basicsr/metrics/fid.py`** 引用了不存在的 `basicsr.models.archs.inception`；
   `Dataset_GaussianDenoising` 里 `from basicsr.utils.scandir import scandir` 路径有误（应为 `basicsr.utils`）。
   两者都在当前流程之外，属潜在问题。
7. `basicsr/train.py` 不传 mask，配 `model_type: AIFlashModel` 会 `KeyError: 'mask'`；请统一用 `my_train.py`。

### 已验证 / 未验证

- ✅ 全部 57 个 Python 文件通过语法编译；包内相对/绝对导入除上述第 1、6 条外均可解析。
- ✅ 11 个 `.pth` 均为 `{'params': ...}` 容器，参数前缀与各自架构匹配。
- ✅ 重写后的 `inference.py`：CLI 参数生效、路径解析、100 张样例图与掩码按名配对、
  tile 分支实参与模型调用、缺失文件报错路径，均通过桩测试（stub torch 的方式静态验证）。
- ⚠️ **未做真实权重的前向/精度实跑**：当前环境未安装 `torch`，
  且第 1 条问题未修复前无法导入 `basicsr.models.archs`。修复后建议先跑
  `python inference.py --input_dir <单张图> --device cpu` 做最小验证。