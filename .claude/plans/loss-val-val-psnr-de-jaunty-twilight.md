# R0C 配置 Loss/Val/可视化 全面分析

## Context

仅针对 `configs/my_aiflash_R0C.yaml` 配置进行分析。R0C 实验的核心目标是验证输入重建约束（`I_low ≈ R × E`），即 Retinex 分解的物理一致性。

---

## 一、R0C 配置 Loss 设计分析

### 1.1 当前损失组合

| 损失 | 类型 | 权重 | 作用区域 | 设计意图 |
|------|------|------|---------|---------|
| `pixel_opt` | CharbonnierLoss_mask | 1.0 | **仅背景** | 背景像素级约束 |
| `input_recon_opt` | CharbonnierLoss | **5.0** | Retinex 约束 | 核心：`R × E ≈ I_low` |
| `smooth_opt` | TVSmoothLoss | 0.1 | env_light | 环境光低频约束 |
| `luminance_opt` | FaceLabLoss | 0.2 | **仅人脸** | 人脸 LAB 色彩 |
| `grad_opt` | GradientLoss | 0.2 | 全图 | 边缘/纹理约束 |
| `perceptual_opt` | PerceptualLoss | 0.0005 | 有 mask 区域 | VGG 特征约束（权重极低） |

### 1.2 设计评估

**合理之处：**
1. `input_recon_opt` 权重 5.0 最高 — 符合 R0C 实验的核心目标（物理一致性约束）
2. `FaceLabLoss` 权重 0.2 — 比主配置的 0.05 更合理，但仍低于背景约束（1.0）
3. `perceptual_opt` 权重 0.0005 极低 — R0C 强调物理约束而非感知质量，合理

**潜在问题：**

**问题 A: 背景/人脸权重仍有差距（严重程度: 中）**
- 背景约束：`pixel_opt` = 1.0
- 人脸约束：`luminance_opt` = 0.2
- 比值：5:1（比主配置的 20:1 好很多，但仍有差距）
- **建议**：如果实验结果显示人脸色彩欠佳，可尝试提升至 0.5

**问题 B: GradientLoss 不使用 mask（严重程度: 低-中）**
- `my_ai_flash_model.py:287` 调用 `self.cri_grad(pred, self.gt)` 未传 mask
- 在人像边缘可能产生梯度伪影
- **影响程度**：R0C 强调物理约束，此问题优先级较低

**问题 C: Val Loss 不完整（严重程度: 中）**
- `_compute_val_losses`（第 469-496 行）只计算：
  - `l_pix`（CharbonnierLoss_mask）
  - `l_input_recon`（输入重建）
  - `l_smooth`（TV 平滑）
- 缺少：`l_illum`、`l_grad`、`l_percep`
- **影响**：
  - `ReduceLROnPlateau` 只基于 `l_pix` 调整学习率
  - 损失曲线图中 train/val 对比不完整
  - 无法监控人脸约束（l_illum）在验证集上的表现

---

## 二、R0C Validation 设置分析

### 2.1 验证集配置（✅ 合理）

R0C 已使用独立验证集：
```yaml
val:
  dataroot_gt: datasets/val_50/gt
  dataroot_lq: datasets/val_50/input
  dataroot_mask: datasets/val_50/mask_personmask
```

- 50 张独立验证集，与训练集（550 张）完全分离
- `val_freq: 1` — 每 epoch 验证一次，频率合理
- `earlystop_patience: 10` — 连续 10 次无改善则停止

### 2.2 PSNR + deltaE 指标组合分析

**现状：**
- PSNR：全图计算（`psnr_ssim.py:97`，mask 参数被忽略）
- deltaE：仅人像区域计算（`psnr_ssim.py:36`，mask 加权）

**代码证据（`psnr_ssim.py:91-97`）：**
```python
# mse = np.mean((img1 - img2)**2)
mask = mask.squeeze(0) if len(mask.shape) == 4 else mask
mask = mask.detach().cpu().numpy().transpose(1,2,0)
mask = mask.astype(np.float64)
# bg=1-mask
# mse = ((img1 - img2)**2 * bg).sum() / (bg.sum() + 1e-8)
mse = np.mean((img1 - img2) ** 2)  # mask 被忽略了
```

**合理性评估：**

| 指标 | 计算范围 | 评估目标 | 合理性 |
|------|---------|---------|--------|
| PSNR | 全图 | 整体结构质量 | ✅ 合理，反映背景+人像综合质量 |
| deltaE | 仅人像 | 人脸色彩准确性 | ✅ 合理，人像任务核心指标 |

**组合使用的合理性：✅ 合理**
- 两个指标互补：一个关注整体结构，一个关注人像色彩
- R0C 实验中，输入重建约束（`input_recon_opt`）主要优化全图一致性，PSNR 可以反映这一约束的效果
- deltaE 专门评估人脸质量，符合人像任务的核心目标

**潜在问题：**
- Early Stopping 逻辑中 PSNR↑ 或 deltaE↓ 任一改善即视为改进
- 当 `input_recon_opt` 优化背景一致性时，PSNR 可能上升但人脸质量未改善
- **建议**：考虑以 deltaE 为主要 Early Stopping 指标

### 2.3 Early Stopping 逻辑（严重程度: 低）

```python
improved = False
if psnr > best_psnr:    # 全图 PSNR
    improved = True
if delta_e < best_deltaE:  # 人像 deltaE
    improved = True
```

**问题：**
- 只保存一份 `best_net_g.pth`
- 当 `best_psnr_iter` 和 `best_deltaE_iter` 不同时，保存的模型可能只对其中一个指标最优

**建议**：
- 分别保存 `best_net_g_psnr.pth` 和 `best_net_g_deltaE.pth`
- 或者明确以 deltaE 作为主指标（人像任务）

---

## 三、R0C 可视化分析

### 3.1 现有机制

| 机制 | 触发位置 | 频率 | 状态 |
|------|---------|------|------|
| TensorBoard 训练 loss | `my_train.py:239` | 每 100 iter | ✅ 正常 |
| TensorBoard 验证 loss/metrics | `my_ai_flash_model.py:601` | 每 epoch | ✅ 正常 |
| 验证图片保存 | `my_ai_flash_model.py:542` | 前 10 张 | ⚠️ 硬编码 |
| 中间张量可视化 | `my_train.py:253` | 每 1 epoch (`vis_freq: 1`) | ✅ 正常 |
| 损失曲线图 | `my_ai_flash_model.py:341` | 每次验证后 | ✅ 正常 |

### 3.2 发现的问题

**问题 A: 缺少人像区域单独可视化（严重程度: 中）**
- 验证时只保存全图（`sr_img` 和 `gt_img`）
- 无法直观评估人脸质量
- **建议**：裁剪人脸区域并排显示

**问题 B: 缺少 deltaE 热力图（严重程度: 中）**
- deltaE 是评估人脸色彩质量的关键指标
- 但无法直观看到误差空间分布
- **建议**：生成 deltaE 热力图叠加在人像上

**问题 C: 中间张量可视化只在训练时生成（严重程度: 低）**
- `my_train.py:253-264` 中间张量可视化只在训练 epoch 结束时触发
- 验证时不会生成 R, E, F, illumination 的可视化
- **影响**：R0C 配置 `vis_freq: 1`，每 epoch 都会触发，此问题影响较小

---

## 四、核心问题回答：R0C 配置下全图 PSNR + 人像 DE 是否合理？

### 4.1 结论：✅ 合理

**理由：**

1. **R0C 实验目标**：验证输入重建约束（`I_low ≈ R × E`），这是全图级别的物理约束
   - 全图 PSNR 可以反映这一约束的效果
   - 如果 `R × E` 与输入 `I_low` 一致，全图 PSNR 应该较高

2. **人像任务核心**：人脸色彩准确性
   - deltaE 在人像区域计算，直接评估人脸质量
   - 与 R0C 的物理约束目标互补

3. **两个指标的分工明确**：
   - PSNR：反映 Retinex 分解的物理一致性（与 `input_recon_opt` 对应）
   - deltaE：反映人像色彩质量（与 `luminance_opt` 对应）

### 4.2 改进建议

**可选改进（非必须）：**

1. **新增人像区域 PSNR**：只在 mask 区域计算 PSNR，与 deltaE 范围一致
   - 优点：可以单独评估人像的像素级质量
   - 缺点：增加指标数量，可能增加选择困难

2. **明确 Early Stopping 主指标**：
   - 当前：PSNR 或 deltaE 任一改善即视为改进
   - 建议：明确以 deltaE 作为主指标（人像任务），PSNR 作为辅助

---

## 五、R0C 配置改进计划

### 阶段 1: 补全 Val Loss（P1）

**文件**：`basicsr/models/my_ai_flash_model.py`

**修改内容**：在 `_compute_val_losses` 中添加缺失的 loss 计算

```python
# 当前只计算 l_pix, l_input_recon, l_smooth
# 需要添加 l_illum, l_grad, l_percep

if self.cri_illum:
    l_illum = self.cri_illum(self.output, self.gt, self.mask)
    losses['l_illum'] = l_illum.item()
    l_total += l_illum.item()

if self.cri_grad:
    l_grad = self.cri_grad(self.output, self.gt)
    losses['l_grad'] = l_grad.item()
    l_total += l_grad.item()

if self.cri_perceptual:
    l_percep = self.cri_perceptual(self.output, self.gt, self.mask)
    losses['l_percep'] = l_percep.item()
    l_total += l_percep.item()
```

**验证**：运行 1 个 epoch，检查 TensorBoard 中 val loss 曲线是否包含所有 loss 项

### 阶段 2: 改进 Early Stopping（P2）

**文件**：`basicsr/my_train.py`

**修改内容**：分别保存 best 模型

```python
# 保存 best PSNR 模型
if psnr > best_psnr:
    best_psnr = psnr
    best_psnr_iter = current_iter
    best_psnr_net_path = os.path.join(opt['path']['models'], 'best_net_g_psnr.pth')
    torch.save({'params': model.net_g.state_dict()}, best_psnr_net_path)

# 保存 best deltaE 模型
if delta_e < best_deltaE:
    best_deltaE = delta_e
    best_deltaE_iter = current_iter
    best_deltae_net_path = os.path.join(opt['path']['models'], 'best_net_g_deltaE.pth')
    torch.save({'params': model.net_g.state_dict()}, best_deltae_net_path)
```

**验证**：检查 `models/` 目录下是否生成了两个 best 模型文件

### 阶段 3: 增强可视化（P3，可选）

**文件**：`basicsr/visualize.py`

**新增功能**：
1. `visualize_deltaE_heatmap` — 生成 deltaE 热力图
2. `visualize_face_crop` — 裁剪人像区域并排显示

**验证**：检查 `visualization/` 目录下是否生成了新类型的可视化图片

---

## 六、关键文件路径

| 文件 | 修改内容 | 优先级 |
|------|---------|--------|
| `basicsr/models/my_ai_flash_model.py` | 补全 `_compute_val_losses` | P1 |
| `basicsr/my_train.py` | 改进 Early Stopping 逻辑 | P2 |
| `basicsr/visualize.py` | 新增 deltaE 热力图和人像裁剪可视化 | P3 |

---

## 七、总结

### R0C 配置评估

| 维度 | 评估 | 说明 |
|------|------|------|
| Loss 设计 | ✅ 合理 | input_recon_opt 权重 5.0 符合实验目标 |
| 验证集 | ✅ 合理 | 使用独立 val_50 |
| 指标组合 | ✅ 合理 | 全图 PSNR + 人像 deltaE 互补 |
| Early Stopping | ⚠️ 可改进 | 建议分别保存 best 模型 |
| Val Loss | ⚠️ 需补全 | 缺少 l_illum, l_grad, l_percep |
| 可视化 | ⚠️ 可增强 | 缺少 deltaE 热力图和人像裁剪 |

### 核心结论

**R0C 配置的全图 PSNR + 人像 deltaE 组合是合理的**，因为：
1. 全图 PSNR 反映 Retinex 分解的物理一致性（与 `input_recon_opt` 对应）
2. 人像 deltaE 反映人脸色彩质量（与 `luminance_opt` 对应）
3. 两个指标分工明确，互补而非冲突

**主要改进建议**：
1. 补全 val loss 计算（P1）
2. 分别保存 best 模型（P2）
3. 可选：新增 deltaE 热力图可视化（P3）
