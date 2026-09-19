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
import matplotlib.colors as mcolors
from torchvision.utils import save_image

# 自定义 colormap：黑 → 暗红 → 红 → 橙 → 黄 → 白
_darkred_to_white = [
    (0.0, (0.0, 0.0, 0.0)),
    (0.2, (0.3, 0.0, 0.0)),    # 暗红
    (0.4, (0.8, 0.0, 0.0)),   # 红
    (0.6, (1.0, 0.5, 0.0)),    # 橙
    (0.8, (1.0, 0.85, 0.0)),  # 黄
    (1.0, (1.0, 1.0, 1.0)),    # 白
]
_cmap_darkred = mcolors.LinearSegmentedColormap.from_list('darkred_to_white', _darkred_to_white)
matplotlib.colormaps.register(_cmap_darkred)


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
    # reflectance_vis = linear_to_srgb(reflectance.clamp(0, 1)).clamp(0, 1) if reflectance is not None else None
    reflectance_vis = reflectance.clamp(0, 1) if reflectance is not None else None
    if reflectance is not None and illumination is not None:
        output_vis = linear_to_srgb((reflectance * illumination).clamp(0, 1)).clamp(0, 1)
    else:
        output_vis = None

    # 设置全局字体大小
    plt.rcParams.update({'font.size': 16})

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Intermediate Visualizations', fontsize=28, fontweight='bold')

    # 定义统一的 title 字体大小
    title_fontsize = 22

    # 统一 colorbar 宽度比例
    cbar_width = "3%"

    def add_colorbar(ax, im):
        """添加统一宽度的 colorbar"""
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=cbar_width, pad=0.05)
        plt.colorbar(im, cax=cax)

    def placeholder_ax(ax, title):
        """占位：无数据时显示灰色背景"""
        ax.set_facecolor('#f0f0f0')
        ax.set_title(title, fontsize=title_fontsize)
        ax.axis('off')

    # Reflectance (R) — sRGB 显示
    ax = axes[0, 0]
    if reflectance_vis is not None:
        img = tensor_to_numpy(reflectance_vis)
        ax.imshow(img)
        ax.set_title('Reflectance (R)', fontsize=title_fontsize)
    else:
        placeholder_ax(ax, 'Reflectance (R)')
    ax.axis('off')

    # Environmental Light (E)
    ax = axes[0, 1]
    if env_light is not None:
        if env_light.shape[-1] == 1 and env_light.shape[-2] == 1:
            # 全局：柱状图
            e_val = env_light.detach().cpu().numpy()
            if e_val.ndim == 4:
                e_val = e_val[0, :, 0, 0]
            colors = ['red', 'green', 'blue']
            ax.bar(colors, e_val, color=colors, alpha=0.7)
            ax.set_title('Environmental Light (E)', fontsize=title_fontsize)
            ax.set_ylim(0, 1)
        else:
            # 空间：热力图（暗红→红→橙→黄→白）
            e_np = env_light[0].detach().cpu().numpy()
            e_gray = e_np.mean(axis=0)
            im = ax.imshow(e_gray, cmap='darkred_to_white', vmin=0, vmax=1 )
            ax.set_title('Environmental Light (E)', fontsize=title_fontsize)
            add_colorbar(ax, im)
    else:
        placeholder_ax(ax, 'Environmental Light (E)')
    ax.axis('off')

    # Flash Map (F)
    ax = axes[0, 2]
    if flash_map is not None:
        f_np = flash_map[0].detach().cpu().numpy()
        f_gray = f_np.mean(axis=0)
        im = ax.imshow(f_gray, cmap='darkred_to_white', vmin=0, vmax=f_gray.max() + 1e-6)
        ax.set_title('Flash Map (F)', fontsize=title_fontsize)
        add_colorbar(ax, im)
    else:
        placeholder_ax(ax, 'Flash Map (F)')
    ax.axis('off')

    # Illumination (E + αF) — 热力图
    ax = axes[1, 0]
    if illumination is not None:
        ill_np = illumination[0].detach().cpu().numpy()
        ill_gray = ill_np.mean(axis=0)
        im = ax.imshow(ill_gray, cmap='darkred_to_white', vmin=0, vmax=ill_gray.max() + 1e-6)
        ax.set_title('Illumination (E + αF)', fontsize=title_fontsize)
        add_colorbar(ax, im)
    else:
        placeholder_ax(ax, 'Illumination (E + αF)')
    ax.axis('off')

    # Output (R × Illumination) — sRGB 显示
    ax = axes[1, 1]
    if output_vis is not None:
        img = tensor_to_numpy(output_vis)
        ax.imshow(img)
        ax.set_title('Output (R × Illum)', fontsize=title_fontsize)
    else:
        placeholder_ax(ax, 'Output (R × Illum)')
    ax.axis('off')

    # GT 对比
    ax = axes[1, 2]
    if gt is not None:
        img = tensor_to_numpy(gt)
        ax.imshow(img)
        ax.set_title('Ground Truth (GT)', fontsize=title_fontsize)
    else:
        ax.set_facecolor('#f0f0f0')
        ax.text(0.5, 0.5, 'No GT', transform=ax.transAxes, fontsize=22,
                ha='center', va='center')
        ax.set_title('Ground Truth (GT)', fontsize=title_fontsize)
    ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出 suptitle 的空间
    save_path = os.path.join(save_dir, f'{prefix}intermediates_{iter_num:06d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 保存单独的图片
    individual_dir = os.path.join(save_dir, 'individual')
    os.makedirs(individual_dir, exist_ok=True)

    def save_heatmap_image(tensor, filepath, cmap='hot', reverse=True):
        """
        将 tensor 保存为热力图图片，右边带 colorbar (带标签、gap)

        Args:
            tensor: (1, C, H, W) 或 (C, H, W) tensor
            filepath: 保存路径
            cmap: matplotlib colormap
            reverse: 是否反转 colormap（True: 数字越小颜色越深）
        """
        if tensor.dim() == 4:
            tensor = tensor[0]  # (C, H, W)
        np_arr = tensor.detach().cpu().numpy()
        if np_arr.ndim == 3:
            np_arr = np_arr.mean(axis=0)  # (H, W)

        h, w = np_arr.shape
        vmin, vmax = 0, np_arr.max() + 1e-6

        # 反转 colormap
        if reverse and not cmap.endswith('_r'):
            cmap = cmap + '_r'

        # 布局参数
        dpi = 100
        gap_px = 5      # gap 宽度（像素）
        label_px = 50   # colorbar + label 宽度（像素）
        total_w = w + gap_px + label_px

        # 创建 figure
        fig = plt.figure(figsize=(total_w / dpi, h / dpi), dpi=dpi)

        # 图像区域
        ax_img = fig.add_axes([0, 0, w / total_w, 1])
        im = ax_img.imshow(np_arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_img.axis('off')

        # colorbar 区域（带 gap）
        cbar_left = (w + gap_px) / total_w
        cbar_width = label_px / total_w
        ax_cbar = fig.add_axes([cbar_left, 0, cbar_width, 1])

        # 自适应字体大小
        font_size = max(8, min(14, h // 30))

        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.ax.tick_params(labelsize=font_size)
        cbar.outline.set_linewidth(0.5)

        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

    def compute_deltaE_map(output, gt):
        """
        计算逐像素的 Delta E 色差图

        Args:
            output: (1, 3, H, W) tensor, sRGB 空间, [0, 1]
            gt: (1, 3, H, W) tensor, sRGB 空间, [0, 1]

        Returns:
            delta_e: (H, W) numpy array, Delta E 值
        """
        import kornia.color as KC
        output_lab = KC.rgb_to_lab(output)  # (1, 3, H, W)
        gt_lab = KC.rgb_to_lab(gt)          # (1, 3, H, W)

        diff_L = output_lab[0, 0] - gt_lab[0, 0]
        diff_a = output_lab[0, 1] - gt_lab[0, 1]
        diff_b = output_lab[0, 2] - gt_lab[0, 2]

        de = torch.sqrt(diff_L ** 2 + diff_a ** 2 + diff_b ** 2 + 1e-8)
        return de.detach().cpu().numpy()  # (H, W)

    # 保存原始图片
    if reflectance_vis is not None:
        save_image(reflectance_vis, os.path.join(individual_dir, f'{prefix}R_{iter_num:06d}.png'))
    if output_vis is not None:
        save_image(output_vis, os.path.join(individual_dir, f'{prefix}output_{iter_num:06d}.png'))
    if gt is not None:
        save_image(gt.clamp(0, 1), os.path.join(individual_dir, f'{prefix}GT_{iter_num:06d}.png'))

    # 保存热力图（图像部分与原图大小一致）
    # 暗红→红→橙→黄→白
    if flash_map is not None:
        save_heatmap_image(flash_map, os.path.join(individual_dir, f'{prefix}F_heatmap_{iter_num:06d}.png'), cmap='darkred_to_white', reverse=False)
    if env_light is not None:
        save_heatmap_image(env_light, os.path.join(individual_dir, f'{prefix}E_heatmap_{iter_num:06d}.png'), cmap='darkred_to_white', reverse=False)
    if illumination is not None:
        save_heatmap_image(illumination, os.path.join(individual_dir, f'{prefix}Illum_heatmap_{iter_num:06d}.png'), cmap='darkred_to_white', reverse=False)

    # 保存色差热力图 (Output vs GT)
    if output_vis is not None and gt is not None:
        delta_e = compute_deltaE_map(output_vis, gt)
        delta_e_path = os.path.join(individual_dir, f'{prefix}deltaE_{iter_num:06d}.png')
        save_heatmap_image(torch.from_numpy(delta_e).unsqueeze(0), delta_e_path, cmap='darkred_to_white', reverse=False)

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
