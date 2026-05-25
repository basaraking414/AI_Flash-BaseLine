# AI Flash Portrait 项目精简计划

## Context

这是项目的初始版本，基于 BasicSR 框架构建。代码中存在大量重复文件和未使用的工具函数，需要在进行重大改进前清理出一个清晰的代码库。

**审查发现**：约 20-25% 的代码是冗余的（~2500-3000行），主要来自复制粘贴的训练脚本、数据加载器和未使用的工具函数。

---

## 一、删除文件清单

### 1.1 根目录（删除 4 个文件）

| 文件 | 原因 |
|------|------|
| `setup.py` | BasicSR 遗留，已被掏空，项目不通过 pip 安装 |
| `setup.cfg` | 代码风格配置，非运行必需 |
| `VERSION` | BasicSR 遗留版本号，无实际用途 |
| `improvement_plan.md` | 已被 `improvement_plan_final.md` 合并替代 |

### 1.2 basicsr/ 训练脚本（删除 2 个，保留 2 个）

| 文件 | 原因 |
|------|------|
| `basicsr/ablation_train.py` | 与 `my_train_new.py` 100% 相同 |
| `basicsr/my_train_new.py` | 与 `train.py` 90%+ 相同，差异仅在 mask 处理 |

**保留**：
- `train.py` — 基础训练脚本
- `my_train.py` — 含 mask 输入 + freeze 策略的训练脚本

### 1.3 basicsr/utils/ 未使用文件（删除 6 个，共 906 行）

| 文件 | 行数 | 原因 |
|------|------|------|
| `utils/flow_util.py` | 180 | 未被任何代码引用 |
| `utils/download_util.py` | 70 | 未被任何代码引用 |
| `utils/bundle_submissions.py` | 107 | 未被任何代码引用 |
| `utils/face_util.py` | 217 | 未被任何代码引用 |
| `utils/create_lmdb.py` | 124 | 仅在 `__init__.py` 导出，实际未调用 |
| `utils/lmdb_util.py` | 208 | 仅被 `create_lmdb.py` 引用 |

### 1.4 configs/ 配置文件（删除 3 个）

| 文件 | 原因 |
|------|------|
| `configs/my_aiflash_v2.yaml` | 中间过渡版本，使用独立数据集路径 |
| `configs/my_aiflash_v4.yaml` | 与 ablation 版本高度相似，定位模糊 |
| `configs/aiflash_new.yaml` | name 字段与 v4 相同，易混淆，仅为超参数变体 |

**保留**：
- `flash.yaml` — 原始 Restormer 基线（无 mask）
- `my_aiflash.yaml` — AIFlash 基础版本
- `my_aiflash_ablation.yaml` — 消融实验（使用不同网络架构）

---

## 二、合并/重构文件

### 2.1 合并 inference 脚本

将 `inference.py` 和 `inference_ablation.py` 合并为一个脚本：

```
inference.py (新版本)
├── 添加 --eval 参数控制是否加载 GT 并计算指标
├── 添加 --arch 参数选择模型架构
├── 添加 --weight 参数指定权重路径
└── 删除硬编码的 start_iter/end_iter/step
```

### 2.2 删除调试代码

- `basicsr/models/archs/my_restormer_arch.py` 末尾的 `visualize_tensor` 函数（约70行）

---

## 三、不修改的文件（Baseline 模型保留）

以下架构文件用于对比实验，保持原样：

- `SwinIR_arch.py` (680行)
- `RetinexFormer_arch.py` (370行)
- `RetinexNet_arch.py` (226行)
- `KinD_arch.py` (214行)
- `ZeroDCE_arch.py` (68行)
- `NAFNet_arch.py` (174行)

---

## 四、本地大文件清理（可选）

这些文件已被 `.gitignore` 排除，但占用大量磁盘空间：

| 目录 | 大小 | 建议 |
|------|------|------|
| `experiments/` | 58 GB | 只保留最新的 1-2 个实验 checkpoint |
| `result/` | 40 GB | 如不再需要可删除 |
| `tb_logger/` | 4.7 MB | 可删除 |

---

## 五、执行步骤

1. **删除冗余文件** — 按 1.1-1.4 清单删除
2. **更新 `utils/__init__.py`** — 移除已删除文件的导入
3. **合并 inference 脚本** — 重构为带参数的统一版本
4. **清理调试代码** — 删除 `visualize_tensor`
5. **提交并推送** — git commit + push

---

## 六、验证

- [ ] 确认 `train.py` 和 `my_train.py` 可正常运行
- [ ] 确认合并后的 `inference.py` 支持 `--eval` 和 `--arch` 参数
- [ ] 确认 `configs/` 中保留的 3 个配置文件可正常加载
- [ ] git status 无意外文件

---

## 预期效果

| 指标 | 精简前 | 精简后 |
|------|--------|--------|
| Python 文件数 | 52 | ~40 |
| 代码行数 | ~11,857 | ~8,500 |
| 训练脚本 | 4 个 | 2 个 |
| 配置文件 | 6 个 | 3 个 |
| inference 脚本 | 2 个 | 1 个 |

---

## 版本演进关系

```
flash.yaml (原始基线)
  |
  |  引入 AIFlashModel + Mask输入 + FaceLabLoss
  v
my_aiflash.yaml (AIFlash 基础版)
  |
  +---> my_aiflash_ablation.yaml  (mask_attention网络, 消融实验)
```

---

## 配置文件核心差异速查

| 配置 | model_type | 网络架构 | 亮度损失权重 | 感知损失权重 |
|------|-----------|---------|-------------|-------------|
| flash.yaml | ImageCleanModel | Restormer | 无 | 0.1 |
| my_aiflash.yaml | AIFlashModel | Restormer_AIFlash | 0.05 | 0.2 |
| my_aiflash_ablation.yaml | AIFlashModel | Restormer_AIFlash_mask_attention | 200 | 0.015 |
