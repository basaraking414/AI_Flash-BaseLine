# Epoch 化训练 + ReduceLROnPlateau 改造计划

## 目标

1. 训练控制从 iteration 改为 epoch，换数据集无需手动调整 iter 参数
2. 学习率调度改为 ReduceLROnPlateau，由 val loss 驱动
3. 所有频率参数（val_freq, print_freq, save_freq, vis_freq）统一为 epoch 单位
4. Loss 曲线 x 轴统一为 epoch

---

## 改动文件清单

### 文件 1: `configs/my_aiflash_R0C.yaml`

所有 iter 参数改为 epoch：

| 之前 | 之后 |
|------|------|
| `iters: [50000]` | 移除 |
| `total_iter: 50000` | `total_epochs: 91` |
| `warmup_iter: -1` | `warmup_epochs: -1` |
| `scheduler: CosineAnnealingRestartCyclicLR periods:[20000,30000]` | `scheduler: ReduceLROnPlateau factor:0.5 patience:5 min_lr:1e-6` |
| `vis_freq: 1000` | `vis_freq: 2` |
| `val_freq: 550` | `val_freq: 1` |
| `print_freq: 100` | `print_freq: 1` |
| `save_checkpoint_freq: 550` | `save_checkpoint_freq: 5` |

### 文件 2: `basicsr/my_train.py`

训练循环从 iter 驱动改为 epoch 驱动：

```
之前: while current_iter <= total_iters:
          while train_data is not None:
              current_iter += 1
              if current_iter > total_iters: break
              optimize_parameters(current_iter)
              update_learning_rate(current_iter, warmup_iter)
              if current_iter % val_freq == 0: validation(...)

之后: for epoch in range(start_epoch, total_epochs):
          while train_data is not None:
              optimize_parameters(current_iter)
              current_iter += 1
              train_data = prefetcher.next()
          # epoch 结束
          model.end_epoch(epoch)
          if (epoch+1) % val_freq == 0:
              val_metric, val_loss = validation(...)
              model.scheduler_step(val_loss['l_pix'])
              # early stopping check
          if (epoch+1) % save_checkpoint_freq == 0:
              model.save(epoch, current_iter)
```

具体改动：
- `create_train_val_dataloader`: 返回 `total_epochs` 和 `num_iter_per_epoch`
- 移除 progressive learning 的 iter groups 逻辑
- `update_learning_rate` 改为每 epoch 调用一次
- 所有 `% print_freq` / `% val_freq` / `% save_checkpoint_freq` 改为 epoch 检查
- warmup 改为 epoch 级别

### 文件 3: `basicsr/models/my_ai_flash_model.py`

- `init_training_settings`:
  - `self.vis_freq` 改为 epoch 频率
  - 移除 `CosineAnnealingRestartCyclicLR` scheduler 创建
  - 新增 `ReduceLROnPlateau` scheduler 创建

- `optimize_parameters`:
  - 移除 vis_freq 可视化检查（移到 epoch 结束处）

- `end_epoch(epoch)`: 参数从 current_iter 改为 epoch

- `record_val_loss(val_loss_dict, epoch)`: 参数改为 epoch

- 新增 `scheduler_step(val_loss)`: 调用 ReduceLROnPlateau.step(val_loss)

- `_save_loss_curves(epoch)`:
  - x 轴改为 "Epoch"
  - 文件名改为 `loss_curve_epoch_{epoch:04d}.png`

- `nondist_validation`: 保存图片命名改为 `_epoch{epoch:04d}.png`

- `setup_optimizers`: 新增 ReduceLROnPlateau 创建

### 文件 4: 不改动

- `basicsr/models/base_model.py` — 框架代码，不动
- `basicsr/models/lr_scheduler.py` — CosineAnnealing 保留（不再使用）
- `basicsr/models/archs/` — 网络结构，无关
- `basicsr/models/losses/` — 损失函数，无关

---

## ReduceLROnPlateau 说明

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5,
    min_lr=1e-6, verbose=True)
```

- `mode='min'`: 监控 loss（越小越好）
- `factor=0.5`: LR 降一半
- `patience=5`: 连续 5 次 val 无改善则降 LR
- `min_lr=1e-6`: LR 下限

每次 validation 后调用 `scheduler.step(val_loss)`，scheduler 自动判断是否需要调整 LR。

---

## 验证清单

- [ ] YAML 中所有参数为 epoch 单位
- [ ] 训练循环按 epoch 迭代
- [ ] Loss 曲线 x 轴为 Epoch
- [ ] Val loss 驱动 ReduceLROnPlateau
- [ ] Early stopping 用 epoch 计数
- [ ] Checkpoint 用 epoch 命名
- [ ] 可视化每 vis_freq 个 epoch 触发
