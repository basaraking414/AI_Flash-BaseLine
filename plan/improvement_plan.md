# AI Flash Portrait Enhancement — 完整改进方案

## 一、背景与目标

**当前状态：**
- CVPR 2026 NTIRE AI Flash Portrait Enhancement Challenge 第 8 名
- 基于 Restormer Transformer，公式：`output = R × (Lenv + Lflash × mask)`
- 当前指标：PSNR 31.17, ΔE 2.83（vs Restormer baseline 29.78/3.68）
- 数据集：600 对图像（550 训练 / 50 测试）

**核心问题：**
1. Retinex 分解缺乏物理约束——R 和 Lenv 可以学到任意分解
2. 方法只做单一任务，缺乏通用性
3. 固定闪光强度，缺乏交互控制
4. 新颖性不够强，实验不充分

**数据设定：**
- 输入 = 无闪光灯的低光人像
- GT = 有闪光灯的人像

---

## 二、核心改进

### 改进 1：输入端自重建约束

**物理模型：**
```
输入（无闪光灯）: Input = R × Lenv                    （精确等式）
GT（有闪光灯）:   GT    = R × (Lenv + Lflash × mask)
```

**损失函数：**
```
L_input_recon = Charbonnier(R × Lenv - Input_linear)
```

- 在 Linear RGB 空间计算（`Input_linear = srgb_to_linear(Input)`）
- 全图计算，权重 0.3~0.5
- 与现有损失（Charbonnier_mask + FaceLab + Gradient + Perceptual）联合训练

**为什么有效：**
- 输入端约束是精确物理等式（输入确实只有环境光）
- R 和 Lenv 同时受输入端和 GT 端两个约束，分解更稳定
- Lflash 的学习被自然隔离到 GT 端
- 这是普通 Retinex 方法没有的优势（它们没有配对的闪光灯 GT）

### 改进 2：可控闪光强度（Alpha 参数）

**公式修改：**
```
illumination = Lenv + alpha × Lflash × soft_mask
output = R × illumination
```

**训练策略：**
- 训练时 alpha ~ Beta(2,2) 分布随机采样（均值 0.5，中间概率高）
- 模型学会在 alpha=0 时"无闪光"（≈ R × Lenv），alpha=1 时"全闪光"
- 验证时固定 alpha=1.0，公平对比

**交互价值：**
- 用户可连续调节补光强度
- alpha=0 → 接近原始低光输入
- alpha=0.5 → 中等闪光
- alpha=1.0 → 全闪光（原始效果）

### 改进 3：即插即用 IDM 模块

**模块设计：**
```python
class IlluminationDecompositionModule(nn.Module):
    """即插即用的光照分解模块"""
    def __init__(self, bottleneck_ch):
        self.env_head = EnvLightHead(bottleneck_ch)      # → Lenv (B,3,1,1)
        self.flash_head = FlashHeadMasked(bottleneck_ch) # → Lflash (B,3,H,W)
        self.ref_head = ConvHead(bottleneck_ch)           # → R (B,3,H,W)

    def forward(self, bottleneck_feat, decoder_feat, mask, alpha=1.0):
        Lenv = self.env_head(bottleneck_feat)
        Lflash = self.flash_head(bottleneck_feat, mask)
        R = torch.sigmoid(self.ref_head(decoder_feat))
        soft_mask = 0.9 * mask + 0.1
        output = R * (Lenv + alpha * Lflash * soft_mask)
        return {'R': R, 'Lenv': Lenv, 'Lflash': Lflash, 'output': output}
```

**即插即用接口约定：**
- 任何 backbone 只需实现 `extract_features(x) → {bottleneck, decoder}`
- 可插入 Restormer、NAFNet、SwinIR 等不同架构

---

## 三、代码修改清单

### 修改 1：`basicsr/models/archs/my_restormer_arch.py`（~20 行）

修改 `Restormer_AIFlash_mask_attention.forward()`：

```python
def forward(self, inp_img, mask, alpha=1.0):
    # ... 所有现有代码不变 ...

    # 改动1: 光照组合加入 alpha
    illumination = env_light_img + alpha * flash_map  # 原为 + flash_map

    # 改动2: 存储中间张量为模型属性（供训练代码访问）
    self._intermediate = {
        'reflectance': reflectance,      # linear RGB
        'env_light': env_light_img,      # linear RGB
        'flash_map': flash_map,          # linear RGB
        'alpha': alpha,
        'illumination': illumination,    # linear RGB
    }

    # 改动3: 返回值不变，保持完全向后兼容
    return out_img  # sRGB, 和原来一样
```

**关键设计决策：** 用 `self._intermediate` 存储中间张量，而非修改 `forward()` 返回值。
原因：现有 3 处代码用 `preds[-1]` 取输出，改动返回值会破坏旧 checkpoint 兼容性。

### 修改 2：`basicsr/models/my_ai_flash_model.py`（~80 行）

#### 2.1 `init_training_settings()` — 注册 input_recon 损失（10 行）

```python
if train_opt.get('input_recon_opt'):
    recon_type = train_opt['input_recon_opt'].pop('type')
    cri_recon_cls = getattr(loss_module, recon_type)
    self.cri_input_recon = cri_recon_cls(**train_opt['input_recon_opt']).to(self.device)
else:
    self.cri_input_recon = None
```

#### 2.2 `feed_train_data()` — 采样 alpha（12 行）

```python
def feed_train_data(self, data):
    self.lq = data['lq'].to(self.device)
    self.mask = data['mask'].to(self.device)
    if 'gt' in data:
        self.gt = data['gt'].to(self.device)

    # 新增: 可控闪光训练 - 随机采样 alpha
    alpha_sampling = self.opt['train'].get('alpha_sampling', False)
    if alpha_sampling and self.is_train:
        self.alpha = float(np.random.beta(2, 2))
    else:
        self.alpha = 1.0
```

#### 2.3 `optimize_parameters()` — 传入 alpha + 计算输入重建损失（30 行）

```python
def optimize_parameters(self, current_iter):
    self.optimizer_g.zero_grad()

    # 改动: 传入 alpha
    preds = self.net_g(self.lq, self.mask, alpha=self.alpha)

    # ... 现有代码不变 ...
    if not isinstance(preds, list):
        preds = [preds]
    self.output = preds[-1]  # 不变，保持兼容

    # === 新增: 输入重建损失 ===
    if self.cri_input_recon and hasattr(self.net_g, '_intermediate'):
        inter = self.net_g._intermediate
        input_linear = srgb_to_linear(self.lq)
        input_recon = inter['reflectance'] * inter['env_light']
        l_input_recon = self.cri_input_recon(input_recon, input_linear)
        loss_dict['l_input_recon'] = l_input_recon
        l_total += l_input_recon

    # ... 后续反向传播代码不变 ...
```

### 修改 3：`configs/my_aiflash_v4.yaml`（~15 行）

```yaml
train:
  # ... 所有现有设置保持不变 ...

  # 新增1: alpha 采样
  alpha_sampling: true

  # 新增2: 输入重建损失
  input_recon_opt:
    type: CharbonnierLoss
    loss_weight: 0.5
    reduction: mean
    eps: !!float 1e-3
```

### 修改 4：验证代码（无需修改）

`nonpad_test()` 和 `nondist_validation()` 不传入 alpha，默认 alpha=1.0。
返回值不变，验证代码不需要任何改动。

---

## 四、实验计划

### Phase 1：验证输入重建约束（3-4 天）

| 组 | 配置 | 预期 |
|----|------|------|
| A | baseline（当前模型，无 L_input_recon） | PSNR 31.17, ΔE 2.83 |
| B | + L_input_recon, weight=0.3 | 预计 +0.3~0.5 dB |
| C | + L_input_recon, weight=0.5 | 预计 +0.5~0.8 dB |
| D | + L_input_recon, weight=1.0 | 需验证是否过高 |

每组训练 ~50K iter（约 8-10 小时/组）

### Phase 2：Alpha 控制消融（2-3 天）

| Variant | Input Recon | Alpha | PSNR | ΔE |
|---------|------------|-------|------|-----|
| Baseline | ✗ | 1.0 固定 | 31.17 | 2.83 |
| +InputRecon | ✓ | 1.0 固定 | ? | ? |
| +Alpha | ✗ | Beta 采样 | 31.17* | 2.83* |
| Full | ✓ | Beta 采样 | ? | ? |

*注：Alpha 训练时随机采样，评估时 α=1.0。如果模型学到不同 α 下的合理输出，α=1.0 时结果应持平或略优。

### Phase 3：新增指标（与 Phase 1 并行，1 天）

- SSIM（代码已有，需启用）
- LPIPS（需安装 `lpips` 包）
- Portrait-region ΔE（用 mask 区域单独计算）

### Phase 4：多架构验证（5-7 天）

| Backbone | 类型 | 参数量 | 用途 |
|----------|------|--------|------|
| Restormer | Transformer | ~1.5M | 当前 baseline |
| NAFNet | CNN | ~4M | 轻量级验证 |
| SwinIR-light | Swin Transformer | ~0.6M | 不同架构验证 |

每个 backbone 做两组实验（原始 vs +IDM），验证通用性。

### Phase 5：效率分析 + 泛化实验（2 天）

- 参数量、FLOPs、推理时间对比
- 不同场景子集的性能分析

---

## 五、Alpha 控制效果验证（定性）

| Alpha 值 | 预期效果 |
|----------|---------|
| 0.0 | 接近原始低光输入（R × Lenv） |
| 0.3 | 微弱补光 |
| 0.5 | 中等闪光 |
| 0.8 | 较强闪光 |
| 1.0 | 全闪光（原始效果） |

用同一张图测试 alpha=[0, 0.25, 0.5, 0.75, 1.0]，可视化过渡效果。

---

## 六、论文定位

### 新标题

"Physically-Constrained Illumination Decomposition for Night-Time Flash Portrait Enhancement"

### 核心贡献

1. **双端物理约束的光照分解框架**：提出输入端自重建约束（`input = R × Lenv`）和 GT 端增强约束（`GT = R × (Lenv + Lflash × mask)`），共享反射率和环境光表示，保证分解的物理一致性。

2. **可控闪光强度**：引入 α 参数实现连续闪光强度调节，训练时 Beta 分布采样让模型学会多强度输出。

3. **即插即用 IDM 模块**：将光照分解逻辑从特定 backbone 解耦为独立模块，在 Restormer、NAFNet、SwinIR 上验证通用性。

4. **系统性实验验证**：消融实验 + 多架构验证 + 效率分析。

### 审稿人质疑预判

| 质疑 | 回应 |
|------|------|
| "自重建约束不是新东西" | 关键区别：有配对闪光灯 GT，输入端和 GT 端同时约束，普通 Retinex 没有 |
| "即插即用只是工程选择" | 在 3 个不同架构上验证通用性，并分析模块学到的不变量 |
| "数据量太小" | 600 对是 NTIRE Challenge 标准数据量，且有跨架构验证 |
| "NTIRE 第 8 名" | 聚焦方法论贡献，排名只是辅助验证 |
| "Alpha 控制的实际价值" | 用户交互式调节是移动端修图的刚需，有明确应用场景 |

---

## 七、投稿策略

### 方案 A：冲 TMM（需补完整实验）

**需要：**
- Phase 1-5 全部完成
- 5+ SOTA 方法全面对比（RetinexNet, NAFNet, KinD, Zero-DCE, Retinexformer, SwinIR, LLFormer）
- 统计显著性检验（配对 t 检验）
- Failure cases 分析

**预计命中率：** 50-60%

### 方案 B：稳妥路线（TCSVT / JVCIR）

**只需：**
- Phase 1 + Phase 2 + Phase 3
- 2-3 个 visual comparison cases

**预计命中率：** 70-80%

### 方案 C：先 Workshop 再 Journal

- 先投 CVPR NTIRE Workshop 扩展版
- 再用完整实验投 TMM/TCSVT

---

## 八、工作量总结

| 任务 | 代码量 | 时间 |
|------|--------|------|
| 修改模型 forward（alpha + 中间张量） | ~20 行 | 1h |
| 修改训练逻辑（alpha 采样 + input_recon loss） | ~80 行 | 2-3h |
| 修改 config | ~15 行 | 0.5h |
| 调试训练 | — | 0.5 天 |
| Phase 1 消融（4 组） | — | 3-4 天 |
| Phase 2 Alpha 消融 | — | 2-3 天 |
| Phase 3 新增指标 | — | 1 天 |
| Phase 4 多架构验证 | — | 5-7 天 |
| Phase 5 效率 + 泛化 | — | 2 天 |
| **总计** | **~115 行** | **~2-3 周** |

---

## 九、验证清单

- [ ] **训练前**：加载旧 checkpoint，确认 forward 返回值不变，PSNR 一致
- [ ] **训练中（100 iter）**：确认 loss 包含 `l_input_recon` 项且数值合理
- [ ] **训练中**：确认 `alpha` 被随机采样，且在 tensorboard 中记录
- [ ] **训练后**：对比新旧模型在 alpha=1.0 时的 PSNR/ΔE
- [ ] **Alpha 控制**：用同一张图测试 alpha=[0, 0.25, 0.5, 0.75, 1.0]，可视化过渡效果
- [ ] **消融**：分别关掉 input_recon 和 alpha_sampling，复现基线对比
- [ ] **多架构**：在 NAFNet 和 SwinIR 上验证 IDM 增量贡献

---

## 十、关键文件

| 文件 | 修改内容 |
|------|---------|
| `basicsr/models/archs/my_restormer_arch.py` | forward 加 alpha 参数 + 存储 _intermediate |
| `basicsr/models/my_ai_flash_model.py` | 注册 loss + alpha 采样 + 计算 L_input_recon |
| `basicsr/models/losses/losses.py` | 可能需新增 SelfReconLoss（或复用 CharbonnierLoss） |
| `configs/my_aiflash_v4.yaml` | 添加 alpha_sampling + input_recon_opt |
| `basicsr/metrics/` | 新增 LPIPS、portrait-ΔE 指标 |
