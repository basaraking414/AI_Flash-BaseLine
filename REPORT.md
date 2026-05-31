# AI Flash Portrait 项目调研报告

> 基于 Retinex 分解的人像闪光重光照系统 — 完整技术分析

---

## 一、项目概述

### 1.1 研究背景与动机

闪光灯人像摄影中，闪光灯的强光会导致人脸过曝、肤色失真、光照不一致等问题。本项目基于 **Retinex 理论**，将输入图像分解为反射率 (Reflectance, R) 和光照 (Illumination, E+F)，通过网络学习环境光 (E) 和闪光光 (F) 的估计，最终合成高质量的闪光人像。

**核心公式**：

```
output = R × (E + α × F)    (在 linear RGB 空间计算)
```

其中：
- **R** — 反射率（场景固有属性，与光照无关）
- **E** — 环境光（ambient illumination）
- **F** — 闪光光（flash illumination，受 mask 引导）
- **α** — 闪光强度控制参数（训练时可随机采样，推理时固定为 1.0）

### 1.2 项目架构总览

- **框架**：基于 [BasicSR](https://github.com/XPixelGroup/BasicSR) 构建
- **主模型**：`AIFlashModel`（`basicsr/models/my_ai_flash_model.py`）
- **主网络**：`Restormer_AIFlash_mask_attention`（`basicsr/models/archs/my_restormer_arch.py`）
- **配置文件**：
  - `flash.yaml` — 基础 Restormer baseline
  - `my_aiflash_ablation.yaml` — 消融实验配置
  - `my_aiflash.yaml` — R0+C 最终版本（当前主配置）
- **实验记录**：19 个实验目录，涵盖从基线到 R0C 变体的完整演进

### 1.3 实验版本演进路线

```
Restormer baseline (flash.yaml)
    ↓
Restormer_AIFlash_v2 (sRGB, 无 mask attention)
    ↓
Restormer_AIFlash_v3-2 (linear RGB, 无 mask attention)
    ↓
Restormer_AIFlash_v4 (linear RGB + mask attention)
    ↓
Illumination_Decomposition_AIFlash (完整 Retinex 分解)
    ↓
R0C_input_recon_constraint v0~v4 (加入输入重建约束 + TV 平滑损失)
```

对比架构实验：RetinexFormer、RetinexNet、NAFNet、KinDNet、ZeroDCE

---

## 二、网络架构分析

### 2.1 整体结构：U-Net + Restormer TransformerBlock

主网络 `Restormer_AIFlash_mask_attention` 采用 **U-Net 编码器-解码器** 结构，以 Restormer 的 TransformerBlock 为基本单元。

**配置参数**（`my_aiflash.yaml`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `inp_channels` | 3 | RGB 输入 |
| `out_channels` | 3 | RGB 输出 |
| `dim` | 16 | 基础通道数 |
| `num_blocks` | [2,2,2,2] | 每级 TransformerBlock 数量 |
| `num_refinement_blocks` | 2 | 精炼模块 Block 数 |
| `heads` | [1,2,4,8] | 每级注意力头数 |
| `ffn_expansion_factor` | 2.66 | GDFN 扩展因子 |
| `LayerNorm_type` | BiasFree | 归一化类型 |

**数据流**：

```
输入 inp_img (B,3,H,W) sRGB
    │
    ▼
srgb_to_linear() → linear RGB
    │
    ▼
OverlapPatchEmbed (3×3 Conv, 3→16)
    │
    ├─── Encoder Level 1: 2× TransformerBlock (dim=16, heads=1)
    │         │
    │         ▼ Downsample (PixelUnshuffle)
    ├─── Encoder Level 2: 2× TransformerBlock (dim=32, heads=2)
    │         │
    │         ▼ Downsample
    ├─── Encoder Level 3: 2× TransformerBlock (dim=64, heads=4)
    │         │
    │         ▼ Downsample
    └─── Latent: 2× TransformerBlock (dim=128, heads=8)
              │
              ├──→ EnvLightHead → env_light (B,3,H/8,W/8)
              │
              └──→ FlashHeadMasked(latent, mask) → flash_map (B,3,H/8,W/8)

Decoder (对称结构):
    Latent → Upsample + concat skip → Level 3 → ... → Level 1
    │
    ▼
Refinement: 2× TransformerBlock
    │
    ▼
Output Conv (3×3) → sigmoid → reflectance (B,3,H,W)
    │
    ▼
illumination = env_light + α × flash_map
output = reflectance × illumination
    │
    ▼
linear_to_srgb() → 最终输出 (B,3,H,W) sRGB
```

### 2.2 TransformerBlock 内部结构

每个 TransformerBlock 包含两个子层：

1. **MDTA (Multi-DConv Head Transposed Self-Attention)**
   - 使用通道注意力（非空间注意力）
   - Q, K, V 通过 1×1 Conv + depthwise 3×1 Conv 投影
   - 注意力计算：`attn = softmax(normalized_K @ normalized_Q^T × temperature)`

2. **GDFN (Gated-Dconv Feed-Forward Network)**
   - 1×1 Conv 扩展 → depthwise 3×3 Conv → GELU 门控 (`x1 × x2`) → 1×1 Conv 压缩

### 2.3 EnvLightHead — 环境光估计模块

**位置**：`basicsr/models/archs/my_restormer_arch.py:192-215`

**结构**：
```
latent (B, 128, H/8, W/8)
    │
    ▼
Conv3×3 (128→128) → ReLU → Conv1×1 (128→3)
    │
    ▼
Gaussian Blur (kernel=15, sigma=5.0)  ← 强制空间平滑
    │
    ▼
sigmoid × 0.9  ← 限制最大值为 0.9
    │
    ▼
env_light (B, 3, H/8, W/8)
```

**设计要点**：
- Gaussian blur 确保环境光空间平滑（物理上环境光是低频信号）
- 0.9 上限防止环境光过亮
- 输出上采样到原图分辨率后与 reflectance 相乘

### 2.4 FlashHeadMasked — 带掩码注意力的闪光估计模块

**位置**：`basicsr/models/archs/my_restormer_arch.py:361-471`

**核心设计**：Cross-Attention 机制，mask 作为 Query，latent 作为 Key/Value

```
mask (B,1,H,W)                    latent (B,128,H/8,W/8)
    │                                  │
    ▼ downsample                       │
mask_low (B,1,H/8,W/8)                │
    │                                  │
    ├──→ q_conv → Q (B, heads, hw, d)  │
    │                                  ├──→ k_conv → K (B, heads, hw, d)
    │                                  ├──→ v_conv → V (B, heads, hw, d)
    │                                  │
    ▼                                  ▼
    attn = softmax(Q @ K^T × scale)
    context = attn @ V
    │
    ▼
Gated Fusion: fused = latent × (1-g) + context × g
    │                                  (g = sigmoid(gate_conv(concat(latent, context))))
    ▼
Conv3×3 → ReLU → Conv3×3 → flash_logits (B,3,H/8,W/8)
    │
    ▼
sigmoid → flash_map
flash_map = flash_map × (0.9 × mask + 0.1)  ← soft mask 调制
```

**设计要点**：
- Cross-attention 让闪光估计依赖于人物位置（mask）和场景内容（latent）
- Soft mask `0.9×mask + 0.1` 确保非人物区域也有微弱闪光（避免硬边界）
- 门控机制控制注意力上下文与原始 latent 的融合比例

### 2.5 Retinex 合成

```python
# 在 my_restormer_arch.py forward() 中
illumination = env_light_img + alpha * flash_map
out_img = reflectance * illumination        # linear RGB 空间
out_img = linear_to_srgb(out_img)           # 转回 sRGB
```

中间结果存储在 `self._intermediate` 字典中，供损失计算和可视化使用：
- `reflectance` — 反射率
- `env_light` — 环境光
- `flash_map` — 闪光图
- `alpha` — 闪光强度参数
- `illumination` — 总光照

---

## 三、损失函数分析

### 3.1 损失函数总览

共 **7 个损失函数**，分为 3 组：

| 损失函数 | 权重 | 作用域 | 功能 |
|---------|------|--------|------|
| `CharbonnierLoss_mask` | 2.0 | 背景区域 (1-mask) | 背景像素重建 |
| `FaceLabLoss` | 1.0 | 面部区域 (mask) | LAB 空间肤色保真 |
| `GradientLoss` | 0.2 | 全局 | 边缘/纹理保持 |
| `ColorLoss` | 0.3 | 全局 | 低频颜色一致性 |
| `PerceptualLoss` (VGG19) | 0.001 | 面部区域 (mask) | 感知质量 |
| `CharbonnierLoss` (input_recon) | 5.0 | 全局 | R×E ≈ 输入约束 |
| `TVSmoothLoss` | 0.1 | 全局 | 环境光空间平滑 |

### 3.2 详细分析

#### CharbonnierLoss_mask (w=2.0)

**位置**：`basicsr/models/losses/losses.py:92-120`

- 公式：`L = mean(sqrt((pred - target)^2 + eps^2) × bg_mask)`
- `bg_mask = 1 - mask`，仅在**背景区域**计算
- 作用：约束非人物区域的重建质量，避免干扰面部优化
- eps = 1e-3

#### FaceLabLoss (w=1.0)

**位置**：`basicsr/models/losses/losses.py:28-89`

- RGB → LAB 转换（使用 `kornia.color.rgb_to_lab`）
- L 通道归一化到 [0,100]，ab 通道归一化到 [-128,128]
- 公式：`de = sqrt(λ_L × (dL/100)^2 + λ_ab × (da/128)^2 + λ_ab × (db/128)^2)`
- 配置：`lambda_l=1.5, lambda_ab=1.0`
- 仅在面部 mask 区域计算，空 mask 时返回零损失（带梯度）

#### GradientLoss (w=0.2)

**位置**：`basicsr/models/losses/losses.py:301-338`

- Sobel 算子提取 x/y 方向梯度
- 公式：`L = L1(sobel(pred), sobel(target))`
- 保持输出图像的边缘结构与 GT 一致

#### ColorLoss (w=0.3)

**位置**：`basicsr/models/losses/losses.py:340-353`

- `AvgPool2d(16)` 降采样后计算 MSE
- 比较低频颜色信息，忽略高频细节
- 公式：`L = MSE(avgpool(pred), avgpool(target))`

#### PerceptualLoss (w=0.001)

**位置**：`basicsr/models/losses/losses.py:355-450`

- VGG19 的 `conv3_4` 和 `conv4_4` 层特征
- 仅在面部区域计算（mask-aware）
- 权重极小（0.001），作为辅助损失
- VGG 权重冻结不训练

#### 输入重建损失 CharbonnierLoss (w=5.0)

**位置**：`basicsr/models/my_ai_flash_model.py:291-298`

- **R0+C 核心约束**：`R × E ≈ srgb_to_linear(input)`
- 权重最高（5.0），确保 Retinex 分解的物理一致性
- 仅在有 `_intermediate` 属性时计算
- 在 linear RGB 空间比较

#### TVSmoothLoss (w=0.1)

**位置**：`basicsr/models/losses/losses.py:452-473`

- Total Variation 正则化
- 公式：`L = |x[:,:,1:,:] - x[:,:,:-1,:]| + |x[:,:,:,1:] - x[:,:,:,:-1]|`
- 约束 env_light 空间平滑（环境光本质上是低频信号）
- 仅在 env_light 有空间维度时计算

### 3.3 损失函数设计逻辑

```
┌─────────────────────────────────────────────────────┐
│                    损失函数设计                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  分区约束：                                          │
│    背景 → CharbonnierLoss_mask (像素重建)            │
│    人脸 → FaceLabLoss + PerceptualLoss (肤色+感知)   │
│                                                     │
│  多空间约束：                                        │
│    sRGB 空间 → CharbonnierLoss                       │
│    LAB 空间  → FaceLabLoss (亮度+色度分离)           │
│    频率空间  → ColorLoss (低频一致性)                │
│    梯度空间  → GradientLoss (边缘保持)               │
│                                                     │
│  物理约束：                                          │
│    输入重建 → R×E ≈ input (Retinex 分解合理性)       │
│                                                     │
│  正则化：                                            │
│    TVSmooth → env_light 空间平滑                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 四、训练流程分析

### 4.1 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | AdamW | weight_decay=1e-4, betas=[0.9, 0.999] |
| 学习率 | 3e-4 | 初始学习率 |
| LR 调度 | CosineAnnealingRestartCyclicLR | periods=[92000, 208000], eta_mins=[0.0003, 1e-5] |
| 梯度裁剪 | clip_grad_norm = 0.01 | 防止梯度爆炸 |
| 总迭代次数 | 300,000 | |
| 批大小 | 4 | R0+C 版本 |
| 输入尺寸 | 256×256 | 随机裁剪 |
| warmup | 无 | warmup_iter = -1 |
| alpha 采样 | 关闭 | 固定 alpha=1.0 |

### 4.2 数据集

- **训练集**：550 张图像（`datasets/train_550/`）
- **验证集**：50 张图像（`datasets/val_50/`）
- **5 种 mask 类型**：
  - `mask_personmask` — 人体分割 mask（主要使用）
  - `mask_hairmask` — 头发区域
  - `mask_skinmask` — 皮肤区域
  - `mask_skinmaskFace` — 面部皮肤
  - `mask_skymask` — 天空区域
- **数据增强**：随机裁剪、翻转、旋转（同步应用于 gt + lq + 所有 mask）
- **尺寸对齐**：确保能被 8 整除（Restormer 4 级下采样）

### 4.3 训练流程

**入口**：`basicsr/my_train.py`

```
main()
  │
  ├─ parse_options() → 解析 YAML 配置
  ├─ 自动检测 resume state → 恢复迭代计数
  ├─ init_loggers() → 文件日志 + TensorBoard
  ├─ create_train_val_dataloader()
  ├─ create_model(opt) → AIFlashModel
  ├─ Early Stopping 初始化 (patience=12)
  │
  └─ 主训练循环:
       while current_iter <= 300000:
           for each batch:
               ├─ 渐进式学习: 根据 iter 选择 batch_size 和 gt_size
               ├─ mask 二值化: mask = (mask > 0).float()
               ├─ model.feed_train_data({lq, gt, mask})
               ├─ model.optimize_parameters(current_iter)
               │    ├─ forward: net_g(lq, mask, alpha)
               │    ├─ 计算 7 个损失
               │    ├─ backward + grad clip
               │    └─ 定期 visualize_intermediates()
               ├─ 每 val_freq (5000) 次迭代: 验证
               │    ├─ PSNR + DeltaE 指标
               │    ├─ 保存最佳模型
               │    ├─ Early stopping 检查
               │    └─ 绘制损失曲线
               └─ end_epoch() → 累积 loss 取均值
```

### 4.4 关键机制

**自动恢复训练**：
- 扫描 `experiments/{name}/training_states/` 目录
- 自动选择最高迭代数的 `.state` 文件
- **注意**：`model.resume_training()` 被注释掉，优化器/调度器状态不恢复

**早停机制**：
- 跟踪 `best_psnr` 和 `best_deltaE`
- 指标改善时重置 `patience_counter`，保存 `best_net_g.pth`
- 连续 12 次验证无改善则停止

**渐进式学习**：
- 支持在不同迭代阶段改变 batch_size 和 patch_size
- 当前配置只有 1 组（无实际渐进）

---

## 五、可视化分析

### 5.1 visualize.py 状态

| 分支 | 状态 | 说明 |
|------|------|------|
| `main` | 完整源码（551 行） | 6 个功能模块 |
| `old-iter-version`（当前） | **源码缺失**，仅 `.pyc` 缓存 | 依赖 `__pycache__/visualize.cpython-313.pyc` |

**风险**：如果 `__pycache__` 被清除（`.gitignore` 会排除），导入将失败：`ModuleNotFoundError: No module named 'basicsr.visualize'`

### 5.2 可视化功能清单

| 函数 | 功能 | 调用位置 | 状态 |
|------|------|---------|------|
| `visualize_intermediates()` | 2×3 网格：R, E, F, Illumination, Output, Stats | `optimize_parameters()` 每 5000 iter | 可能未生效 |
| `save_loss_curve()` | 从日志文件绘制损失曲线 | 未被调用 | 死代码 |
| `create_alpha_comparison()` | 对比不同 alpha 值的输出 | 未被调用 | 死代码 |
| `visualize_deltaE_heatmap()` | DeltaE 热力图 | 验证流程 | 已使用 |
| `visualize_face_crop()` | 面部区域裁剪可视化 | 未被调用 | 死代码 |
| `_save_loss_curves()` | 从内存历史绘制损失曲线 | 每次验证后 | **实际使用** |

### 5.3 中间张量可视化详情

`visualize_intermediates()` 创建 2×3 网格：

| 位置 | 内容 | 说明 |
|------|------|------|
| (0,0) | Reflectance (R) | sRGB 转换后显示，含 mean/std |
| (0,1) | Environmental Light (E) | 空间分布用热力图，全局用柱状图 |
| (0,2) | Flash Map (F) | grayscale heatmap + colorbar |
| (1,0) | Illumination (E+αF) | 归一化热力图 |
| (1,1) | Output (R×Illum) | sRGB 重建结果 |
| (1,2) | GT 或统计信息 | 可选 GT 对比 |

同时保存单独图片到 `individual/` 子目录（R, F, output, GT）。

### 5.4 问题：可视化可能未生效

检查所有 19 个实验目录，**未发现** `visualizations/` 子目录的输出文件。可能原因：
- `vis_freq=5000` 导致间隔过长
- 训练在触发前终止
- `_intermediate` 属性未正确设置

---

## 六、问题与 Bug 清单

### 6.1 严重问题

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | `visualize.py` 源码在当前分支缺失 | `basicsr/visualize.py` | 无法修改或调试可视化功能；`__pycache__` 清除后训练崩溃 |
| 2 | `model.resume_training()` 被注释掉 | `my_train.py:180` | 恢复训练时优化器/调度器状态丢失，学习率重置到初始值 |
| 3 | `boundary_mask` 函数缺少 `self` 参数 | `my_ai_flash_model.py:429` | 作为类方法定义但缺少 self，且从未被调用（死代码 + 语法错误） |

### 6.2 中等问题

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 4 | `nondist_validation` 限制保存前 10 张图 | `my_ai_flash_model.py:548` | 无法全面评估验证集效果 |
| 5 | `create_alpha_comparison` 从未调用 | `visualize.py` | 死代码，无法展示可控闪光效果 |
| 6 | 验证集直接使用训练集（部分配置） | `flash.yaml` | 验证指标不可靠，可能过拟合 |

### 6.3 代码质量问题

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 7 | `calculate_psnr` 中 mask 参数被声明但未使用 | `metrics/psnr_ssim.py` | PSNR 计算实际未做 mask 过滤 |
| 8 | `FeedForward` 中 `groups` 参数可能不正确 | `my_restormer_arch.py:85` | depthwise conv 分组数 = hidden_features×2 |
| 9 | `FlashLightHead` 类未被主架构使用 | `my_restormer_arch.py:217-228` | 死代码 |
| 10 | `nonpad_test` 不传入 alpha 参数 | `my_ai_flash_model.py:454` | 推理时 alpha 默认 1.0，但未显式传递 |

---

## 七、改进建议

### 7.1 代码修复（优先级高）

1. **恢复 `visualize.py` 源码**：`git checkout main -- basicsr/visualize.py`
2. **恢复 `model.resume_training()` 调用**：取消 `my_train.py:180` 的注释
3. **修复 `boundary_mask`**：添加 `@staticmethod` 装饰器或 `self` 参数
4. **修复 `calculate_psnr`**：使其正确使用 mask 过滤

### 7.2 功能增强

1. 移除 `nondist_validation` 的 10 张限制，改为可配置参数
2. 集成 `create_alpha_comparison` 到训练结束后的自动执行流程
3. 添加 SSIM 指标到验证流程
4. 启用 WandB 日志集成（当前 `project: ~`）

### 7.3 训练优化

1. 启用学习率 warmup（当前 `warmup_iter: -1`）
2. 尝试更大输入尺寸（当前 256×256，原图 768×768）
3. 增加训练数据量（当前仅 550 张）
4. 启用混合精度训练（`use_amp` 已实现但默认关闭）

### 7.4 架构改进

1. 在 FlashHeadMasked 中添加位置编码（positional encoding）
2. 探索 specular highlight 估计（`use_specular=True` 已预留接口）
3. 尝试多尺度 FlashHead（在不同解码层级估计闪光）
4. 探索视频闪光 relighting 的时序一致性

---

## 附录：实验对比结果

| 方法 | PSNR ↑ | DeltaE ↓ | 说明 |
|------|--------|----------|------|
| Restormer (baseline) | 30.21 | 3.33 | 无 Retinex 分解 |
| RetinexNet | 26.78 | 5.04 | 早期 Retinex 网络 |
| ZeroDCE | 28.00 | 4.17 | 零参考增强 |
| NAFNet | 28.07 | 4.44 | Nonlinear Activation Free |
| KinDNet | 29.17 | 3.65 | 基于 Retinex 的分解网络 |
| RetinexFormer | 30.73 | 3.08 | Transformer + Retinex |
| **AIFlash (ours)** | **31.17** | **2.83** | **Retinex + mask attention** |

最佳模型：`experiments/R0C_input_recon_constraint_v4/models/best_net_g.pth`（285000 iters）
