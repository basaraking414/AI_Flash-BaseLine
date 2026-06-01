"""
AI Flash Portrait 可视化工具

功能：
1. 训练过程中可视化 R, E, F, illumination 等中间张量
2. 绘制训练损失曲线
3. 对比不同 alpha 值的输出效果

使用方法：
    from basicsr.visualize import visualize_intermediates, save_loss_curve, create_alpha_comparison
"""

import os
import re
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image


# ============================================================
# 1. 中间张量可视化
# ============================================================

def tensor_to_numpy(tensor):
    """将 tensor 转换为可显示的 numpy 数组 (H, W, 3)"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def visualize_intermediates(intermediates, save_dir, iter_num, prefix='', gt=None):
    """
    可视化并保存中间张量

    Args:
        intermediates: dict, 包含 R, E, F, illumination 等张量（linear RGB 空间）
        save_dir: str, 保存目录
        iter_num: int, 当前迭代次数
        prefix: str, 文件名前缀
        gt: tensor, GT 图像 (1, 3, H, W) sRGB 空间，可选

    Returns:
        str: 保存的图片路径
    """
    os.makedirs(save_dir, exist_ok=True)

    from basicsr.models.archs.my_restormer_arch import linear_to_srgb

    reflectance = intermediates.get('reflectance')
    env_light = intermediates.get('env_light')
    flash_map = intermediates.get('flash_map')
    illumination = intermediates.get('illumination')
    alpha = intermediates.get('alpha', 1.0)

    # linear RGB → sRGB（用于可视化显示）
    reflectance_vis = linear_to_srgb(reflectance.clamp(0, 1)).clamp(0, 1) if reflectance is not None else None
    if reflectance is not None and illumination is not None:
        output_vis = linear_to_srgb((reflectance * illumination).clamp(0, 1)).clamp(0, 1)
    else:
        output_vis = None

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Intermediate Visualizations (Iter {iter_num})', fontsize=16)

    # Reflectance (R) — sRGB 显示
    if reflectance_vis is not None:
        ax = axes[0, 0]
        img = tensor_to_numpy(reflectance_vis)
        ax.imshow(img)
        ax.set_title(f'Reflectance (R, sRGB)\nmean={reflectance.mean():.3f}, std={reflectance.std():.3f}')
        ax.axis('off')

    # Environmental Light (E)
    if env_light is not None:
        ax = axes[0, 1]
        if env_light.shape[-1] == 1 and env_light.shape[-2] == 1:
            # 全局：柱状图
            e_val = env_light.detach().cpu().numpy()
            if e_val.ndim == 4:
                e_val = e_val[0, :, 0, 0]
            colors = ['red', 'green', 'blue']
            ax.bar(colors, e_val, color=colors, alpha=0.7)
            ax.set_title(f'Environmental Light (E)\nRGB=[{e_val[0]:.3f}, {e_val[1]:.3f}, {e_val[2]:.3f}]')
            ax.set_ylim(0, 1)
        else:
            # 空间：热力图
            img = tensor_to_numpy(env_light)
            img = img / (img.max() + 1e-6)
            ax.imshow(img)
            ax.set_title(f'Environmental Light (E)\nmean={env_light.mean():.3f}')
            ax.axis('off')

    # Flash Map (F)
    if flash_map is not None:
        ax = axes[0, 2]
        f_np = flash_map[0].detach().cpu().numpy()
        f_gray = f_np.mean(axis=0)
        im = ax.imshow(f_gray, cmap='hot', vmin=0, vmax=f_gray.max() + 1e-6)
        ax.set_title(f'Flash Map (F)\nmean={flash_map.mean():.3f}, max={flash_map.max():.3f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Illumination (E + αF)
    if illumination is not None:
        ax = axes[1, 0]
        img = tensor_to_numpy(illumination)
        img = img / (img.max() + 1e-6)
        ax.imshow(img)
        ax.set_title(f'Illumination (E + αF)\nα={alpha:.2f}, mean={illumination.mean():.3f}')
        ax.axis('off')

    # Output (R × Illumination) — sRGB 显示
    if output_vis is not None:
        ax = axes[1, 1]
        img = tensor_to_numpy(output_vis)
        ax.imshow(img)
        ax.set_title(f'Output (R × Illum, sRGB)\nmean={output_vis.mean():.3f}')
        ax.axis('off')

    # GT 对比
    ax = axes[1, 2]
    if gt is not None:
        img = tensor_to_numpy(gt)
        ax.imshow(img)
        ax.set_title('Ground Truth (GT)')
    else:
        ax.text(0.5, 0.5, 'No GT', transform=ax.transAxes, fontsize=14,
                ha='center', va='center')
    ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{prefix}intermediates_{iter_num:06d}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    # 保存单独的图片（R 和 output 转 sRGB 保存）
    individual_dir = os.path.join(save_dir, 'individual')
    os.makedirs(individual_dir, exist_ok=True)
    if reflectance_vis is not None:
        save_image(reflectance_vis, os.path.join(individual_dir, f'{prefix}R_{iter_num:06d}.png'))
    if flash_map is not None:
        save_image(flash_map.clamp(0, 1), os.path.join(individual_dir, f'{prefix}F_{iter_num:06d}.png'))
    if output_vis is not None:
        save_image(output_vis, os.path.join(individual_dir, f'{prefix}output_{iter_num:06d}.png'))
    if gt is not None:
        save_image(gt.clamp(0, 1), os.path.join(individual_dir, f'{prefix}GT_{iter_num:06d}.png'))

    return save_path


# ============================================================
# 2. 损失曲线绘制
# ============================================================

def save_loss_curve(log_file, save_path, window_size=100):
    """
    从日志文件中读取损失数据并绘制曲线

    Args:
        log_file: str, 日志文件路径 (TensorBoard events 或文本日志)
        save_path: str, 保存图片路径
        window_size: int, 滑动平均窗口大小
    """
    # 尝试从 TensorBoard 读取
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_acc = EventAccumulator(os.path.dirname(log_file))
        event_acc.Reload()

        losses = {}
        for tag in event_acc.Tags()['scalars']:
            if tag.startswith('l_') or tag == 'psnr':
                events = event_acc.Scalars(tag)
                losses[tag] = {
                    'steps': [e.step for e in events],
                    'values': [e.value for e in events]
                }
        if losses:
            _plot_losses(losses, save_path, window_size)
            return save_path
    except Exception as e:
        print(f"TensorBoard read failed: {e}, trying text log...")

    # 尝试读取文本日志
    try:
        losses = _parse_text_log(log_file)
        if losses:
            _plot_losses(losses, save_path, window_size)
            return save_path
    except Exception as e:
        print(f"Text log read failed: {e}")

    print("No loss data found")
    return None


def _parse_text_log(log_file):
    """解析 BasicSR 文本日志"""
    losses = {}
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'iter:\s*(\d+)', line)
            if match:
                iter_num = int(match.group(1))
                for loss_match in re.finditer(r'(l_\w+):\s*([\d.]+)', line):
                    name = loss_match.group(1)
                    value = float(loss_match.group(2))
                    if name not in losses:
                        losses[name] = {'steps': [], 'values': []}
                    losses[name]['steps'].append(iter_num)
                    losses[name]['values'].append(value)
    return losses


def _plot_losses(losses, save_path, window_size=100):
    """绘制损失曲线"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    main_losses = {k: v for k, v in losses.items() if k in ['l_pix', 'l_total']}
    other_losses = {k: v for k, v in losses.items() if k not in ['l_pix', 'l_total', 'psnr']}

    # 主损失
    ax = axes[0]
    for name, data in main_losses.items():
        steps, values = data['steps'], data['values']
        if len(values) > window_size:
            smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            ax.plot(steps[window_size-1:], smoothed, label=f'{name} (smoothed)', linewidth=2)
        ax.plot(steps, values, alpha=0.3, label=name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Main Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 其他损失
    ax = axes[1]
    for name, data in other_losses.items():
        steps, values = data['steps'], data['values']
        if len(values) > window_size:
            smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            ax.plot(steps[window_size-1:], smoothed, label=f'{name} (smoothed)', linewidth=2)
        ax.plot(steps, values, alpha=0.3, label=name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Other Losses (l_input_recon, l_grad, l_percep, etc.)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ============================================================
# 3. Alpha 控制效果对比
# ============================================================

def create_alpha_comparison(model, input_tensor, mask_tensor, alphas=None, save_path=None):
    """
    对比不同 alpha 值的输出效果

    Args:
        model: 训练好的模型
        input_tensor: 输入图像 (1, 3, H, W)
        mask_tensor: 掩膜 (1, 1, H, W)
        alphas: alpha 值列表
        save_path: 保存路径
    """
    if alphas is None:
        alphas = [0.0, 0.3, 0.5, 0.7, 1.0]

    model.eval()
    fig, axes = plt.subplots(2, len(alphas), figsize=(4 * len(alphas), 8))

    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            output = model(input_tensor, mask_tensor, alpha=alpha)
            inter = model._intermediate

            ax = axes[0, i]
            img = tensor_to_numpy(output)
            ax.imshow(img)
            ax.set_title(f'α = {alpha}')
            ax.axis('off')

            ax = axes[1, i]
            flash = inter['flash_map'][0].detach().cpu().numpy().mean(axis=0)
            ax.imshow(flash, cmap='hot')
            ax.set_title(f'Flash (α={alpha})')
            ax.axis('off')

    plt.suptitle('Alpha Control Comparison', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    plt.show()
    return None


# ============================================================
# 4. 快速测试
# ============================================================

if __name__ == '__main__':
    # 测试可视化功能
    print("Testing visualization functions...")

    # 创建模拟数据
    intermediates = {
        'reflectance': torch.rand(1, 3, 64, 64),
        'env_light': torch.rand(1, 3, 1, 1) * 0.5,
        'flash_map': torch.rand(1, 3, 64, 64) * 0.3,
        'illumination': torch.rand(1, 3, 64, 64) * 0.8,
        'alpha': 0.7,
    }

    # 测试中间张量可视化
    save_path = visualize_intermediates(intermediates, './test_viz', iter_num=100)
    print(f"Saved intermediates visualization: {save_path}")

    # 测试损失曲线（创建模拟日志）
    os.makedirs('./test_viz', exist_ok=True)
    with open('./test_viz/test_log.txt', 'w') as f:
        for i in range(1000):
            f.write(f"iter: {i*100}, l_pix: {0.5 * np.exp(-i/500) + np.random.randn()*0.01}, "
                    f"l_input_recon: {0.3 * np.exp(-i/300) + np.random.randn()*0.005}, "
                    f"l_grad: {0.1 + np.random.randn()*0.002}\n")

    save_path = save_loss_curve('./test_viz/test_log.txt', './test_viz/loss_curve.png')
    print(f"Saved loss curve: {save_path}")

    print("Test completed!")
