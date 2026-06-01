# Validation Loss Curve + Early Stopping Design

## Problem

Training loss curves are based on per-iteration single-image loss, causing heavy noise and oscillation. No early stopping mechanism exists — training runs for fixed `total_iter` regardless of overfitting.

## Goals

1. Plot loss curves based on validation set averages (smooth, meaningful)
2. Stop training early when validation metrics plateau
3. Automatically save the best model checkpoint

## Non-Goals

- Modifying the training loop structure beyond what's needed
- Adding new loss functions or metrics
- Distributed training changes

---

## Design

### 1. Validation Loss Computation

**File:** `basicsr/models/my_ai_flash_model.py` → `nondist_validation()`

During the existing validation loop (which already runs inference on each val image), also compute and accumulate losses:

- Reuse `self.net_g._intermediate` from the inference pass (already populated by `nonpad_test`)
- Note: `nonpad_test` calls `self.net_g(img, mask)` without alpha → defaults to alpha=1.0
- Run the same loss functions used in training: `cri_pix`, `cri_input_recon`, `cri_smooth`
- Accumulate per-image loss values, divide by `cnt` at the end
- Return both `current_metric` (existing) and `val_loss_dict` (new)

Return value change:
```python
# Before:
return current_metric

# After:
return current_metric, val_loss_dict
```

Where `val_loss_dict` contains: `{'l_pix': avg, 'l_input_recon': avg, 'l_smooth': avg, 'l_total': avg}`

### 2. Early Stopping Logic

**File:** `basicsr/my_train.py` → training loop

State variables:
```python
best_psnr = -float('inf')
best_deltaE = float('inf')
patience_counter = 0
patience = opt['train'].get('earlystop_patience', 12)
```

After each validation:
```python
current_metric, val_loss_dict = model.validation(...)

# Parse metrics by key name (not by order)
psnr = model.metric_results.get('psnr', -float('inf'))
delta_e = model.metric_results.get('deltaE', float('inf'))

improved = False
if psnr > best_psnr:
    best_psnr = psnr
    improved = True
if delta_e < best_deltaE:
    best_deltaE = delta_e
    improved = True

if improved:
    patience_counter = 0
    # Save best model
    model.save_network(model.net_g, 'best_net_g', current_iter)
    # Save best metrics JSON
    save_best_metrics(best_psnr, best_deltaE, current_iter)
else:
    patience_counter += 1
    if patience_counter >= patience:
        logger.info(f'Early stopping at iter {current_iter} (no improvement for {patience} validations)')
        break
```

### 3. Best Model Saving

**File:** `basicsr/my_train.py`

On metric improvement:
- Save `best_net_g.pth` (overwrites previous best)
- Save `best_metrics.json`:
  ```json
  {
    "best_psnr": 24.56,
    "best_deltaE": 3.21,
    "best_iter": 15000,
    "timestamp": "2026-05-26T10:30:00"
  }
  ```

### 4. Loss Curve Redesign

**File:** `basicsr/models/my_ai_flash_model.py` → `_save_loss_curves()`

Two data sources:
- `self.loss_history`: per-iteration train loss (kept for reference)
- `self.val_loss_history`: per-validation averaged loss (new)

Plot layout:
- **Top subplot:** `l_pix` — val points (thick line + markers) + train raw (thin, alpha=0.2)
- **Bottom subplot:** Other losses (`l_input_recon`, `l_smooth`, etc.) — same style

Bug fix: move `plt.savefig(overview_path)` before `plt.close()`.

### 5. YAML Config Changes

```yaml
train:
  earlystop_patience: 12  # 新增

val:
  val_freq: !!float 2e3  # 已有
```

---

## Files Modified

| File | Changes |
|------|---------|
| `basicsr/models/my_ai_flash_model.py` | `nondist_validation` 累计 val loss；`_save_loss_curves` 支持 val+train 双曲线；修复 bug |
| `basicsr/my_train.py` | 捕获 val 返回值；early stopping 逻辑；best model 保存 |
| `configs/my_aiflash_R0C.yaml` | 新增 `earlystop_patience: 12` |

## Verification

- [ ] Val loss 在日志中输出（`val_l_pix`, `val_l_total` 等）
- [ ] Loss 曲线包含 val 均值点（平滑）+ train 细线（参考）
- [ ] Early stopping 在 12 次 val 无改善后触发
- [ ] `best_net_g.pth` 和 `best_metrics.json` 在指标改善时保存
- [ ] `_save_loss_curves` 不再报错（bug 修复）
