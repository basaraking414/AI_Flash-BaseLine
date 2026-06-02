"""
AI Flash Portrait — 完整推理评估脚本

功能：
1. 指定模型结构 + 权重路径
2. 推理 + 中间张量可视化（R, E, F, illumination）
3. 计算所有指标：PSNR, DeltaE, SSIM
4. 计算所有 loss（与训练一致的 6 个 loss）
5. 效率分析：FLOPs, 参数量, 推理时间
6. 输出汇总报告（JSON + 终端打印）

用法：
    python run_inference.py --config configs/inference.yaml
    python run_inference.py --config configs/inference.yaml --weights path/to/model.pth
"""

import argparse
import json
import os
import time
from collections import OrderedDict
from copy import deepcopy
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from natsort import natsorted

from basicsr.models.archs import define_network
from basicsr.models.archs.my_restormer_arch import srgb_to_linear
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_deltaE, calculate_ssim
from basicsr.visualize import visualize_intermediates


# ============================================================
# 1. 配置加载
# ============================================================

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# ============================================================
# 2. 模型构建与加载
# ============================================================

def build_model(config):
    """根据 config.network_g 动态实例化模型"""
    net_cfg = deepcopy(config['network_g'])
    model = define_network(net_cfg)
    return model


def load_weights(model, weights_path, device):
    """加载 checkpoint，支持 params/state_dict/raw 格式"""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f'Weights not found: {weights_path}')

    ckpt = torch.load(weights_path, map_location=device)

    # 提取 state_dict
    if isinstance(ckpt, dict):
        if 'params' in ckpt:
            state_dict = ckpt['params']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # 去除 module. 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    print(f'Loaded weights from: {weights_path}')
    return model


# ============================================================
# 3. 效率分析
# ============================================================

def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6,
    }


def compute_flops(model, input_size=(1, 3, 256, 256), device='cuda'):
    """计算 FLOPs（使用 thop）"""
    try:
        from thop import profile
        model_for_flops = deepcopy(model).to(device).eval()
        dummy_input = torch.randn(*input_size).to(device)
        dummy_mask = torch.ones(input_size[0], 1, input_size[2], input_size[3]).to(device)
        flops, params = profile(model_for_flops, inputs=(dummy_input, dummy_mask), verbose=False)
        del model_for_flops, dummy_input, dummy_mask
        torch.cuda.empty_cache()
        return {
            'flops': flops,
            'flops_G': flops / 1e9,
            'params_from_thop': params,
        }
    except ImportError:
        print('Warning: thop not installed. Skipping FLOPs calculation.')
        print('Install with: pip install thop')
        return {'flops': 0, 'flops_G': 0, 'params_from_thop': 0}
    except Exception as e:
        print(f'Warning: FLOPs calculation failed: {e}')
        return {'flops': 0, 'flops_G': 0, 'params_from_thop': 0}


def measure_inference_time(model, input_size, device, warmup=3, runs=10):
    """测量推理时间（含 GPU 同步）"""
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    dummy_mask = torch.ones(input_size[0], 1, input_size[2], input_size[3]).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input, dummy_mask)
            torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.time()
            model(dummy_input, dummy_mask)
            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # ms

    del dummy_input, dummy_mask
    torch.cuda.empty_cache()

    times = np.array(times)
    return {
        'avg_ms': float(times.mean()),
        'std_ms': float(times.std()),
        'min_ms': float(times.min()),
        'max_ms': float(times.max()),
        'warmup_runs': warmup,
        'timed_runs': runs,
    }


# ============================================================
# 4. 数据加载
# ============================================================

def get_image_files(directory):
    """获取目录下所有图片文件"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(directory, ext)))
    return natsorted(files)


def load_img(filepath):
    """加载图片，返回 [0,1] float tensor (1, 3, H, W)"""
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {filepath}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return img_tensor


def load_mask(filepath):
    """加载 mask，返回 [0,1] float tensor (1, 1, H, W)"""
    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'Cannot read mask: {filepath}')
    mask = (mask > 0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return mask_tensor


def pad_to_multiple(tensor, multiple=8):
    """将 tensor 的 H, W pad 到 multiple 的倍数"""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, h, w


def save_img(tensor, filepath):
    """保存 tensor 为图片"""
    img = tensor.squeeze(0).clamp(0, 1).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, img)


# ============================================================
# 5. 推理
# ============================================================

def run_inference(model, lq_tensor, mask_tensor, device, alpha=1.0):
    """单张图推理"""
    lq_tensor = lq_tensor.to(device)
    mask_tensor = mask_tensor.to(device)

    with torch.no_grad():
        output = model(lq_tensor, mask_tensor, alpha=alpha)

    intermediate = getattr(model, '_intermediate', None)
    return output, intermediate


# ============================================================
# 6. 评估指标
# ============================================================

def compute_metrics(output, gt, mask):
    """计算 PSNR, DeltaE, SSIM"""
    results = {}
    results['psnr'] = calculate_psnr(output, gt, mask)
    results['deltaE'] = calculate_deltaE(output, gt, mask).item()
    results['ssim'] = calculate_ssim(output, gt, mask)
    return results


def compute_all_losses(output, gt, mask, lq, intermediate, device):
    """计算所有训练 loss"""
    from basicsr.models.losses.losses import (
        CharbonnierLoss_mask, FaceLabLoss, GradientLoss,
        PerceptualLoss, CharbonnierLoss, TVSmoothLoss
    )

    losses = {}

    # l_pix: CharbonnierLoss_mask (背景区域)
    cri_pix = CharbonnierLoss_mask(loss_weight=1.0, eps=1e-3).to(device)
    losses['l_pix'] = cri_pix(output, gt, mask).item()

    # l_illum: FaceLabLoss (面部区域)
    cri_illum = FaceLabLoss(loss_weight=1.0, lambda_l=1.5, lambda_ab=1.0).to(device)
    losses['l_illum'] = cri_illum(output, gt, mask).item()

    # l_grad: GradientLoss (全局)
    cri_grad = GradientLoss(loss_weight=1.0).to(device)
    losses['l_grad'] = cri_grad(output, gt).item()

    # l_percep: PerceptualLoss (面部区域)
    cri_percep = PerceptualLoss(
        layer_weights={'conv3_4': 1.0, 'conv4_4': 1.0},
        vgg_type='vgg19', use_input_norm=True, loss_weight=1.0
    ).to(device)
    losses['l_percep'] = cri_percep(output, gt, mask).item()

    # l_input_recon: CharbonnierLoss (R*E vs input_linear)
    if intermediate is not None:
        cri_recon = CharbonnierLoss(loss_weight=1.0, eps=1e-3).to(device)
        input_recon = intermediate['reflectance'] * intermediate['env_light']
        input_linear = srgb_to_linear(lq)
        losses['l_input_recon'] = cri_recon(input_recon, input_linear).item()

        # l_smooth: TVSmoothLoss (env_light)
        cri_smooth = TVSmoothLoss(loss_weight=1.0).to(device)
        env_light = intermediate['env_light']
        if env_light.shape[-1] > 1:
            losses['l_smooth'] = cri_smooth(env_light).item()
        else:
            losses['l_smooth'] = 0.0
    else:
        losses['l_input_recon'] = 0.0
        losses['l_smooth'] = 0.0

    losses['l_total'] = sum(losses.values())
    return losses


# ============================================================
# 7. 可视化
# ============================================================

def visualize_single(intermediate, save_dir, img_name, gt=None):
    """调用 visualize_intermediates 保存中间张量可视化"""
    if intermediate is not None:
        visualize_intermediates(
            intermediate, save_dir, iter_num=0, prefix=f'{img_name}_', gt=gt
        )


# ============================================================
# 8. 主流程
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='AI Flash Portrait Inference')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    parser.add_argument('--weights', type=str, default=None, help='Override weights path')
    parser.add_argument('--device', type=str, default=None, help='Override device')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    device = args.device or config.get('device', 'cuda:0')
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    weights_path = args.weights or config['weights_path']
    output_dir = config.get('output_dir', 'results/inference')
    pad_multiple = config.get('pad_multiple', 8)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    print('=' * 60)
    print('AI Flash Portrait — Inference Evaluation')
    print('=' * 60)

    # 2. 构建模型 + 加载权重
    print(f'\n[1/5] Building model: {config["network_g"]["type"]}')
    model = build_model(config).to(device)
    model.eval()
    model = load_weights(model, weights_path, device)

    # 3. 效率分析
    print('\n[2/5] Efficiency analysis...')
    params_info = count_parameters(model)
    print(f'  Total params: {params_info["total_M"]:.2f}M')
    print(f'  Trainable params: {params_info["trainable_M"]:.2f}M')

    eff_cfg = config.get('efficiency', {})
    flops_info = {'flops': 0, 'flops_G': 0}
    time_info = {'avg_ms': 0, 'std_ms': 0}

    if eff_cfg.get('compute_flops', False):
        input_size = tuple(eff_cfg.get('input_size', [1, 3, 256, 256]))
        flops_info = compute_flops(model, input_size, device)
        print(f'  FLOPs: {flops_info["flops_G"]:.2f}G')

    if eff_cfg.get('compute_time', False):
        input_size = tuple(eff_cfg.get('input_size', [1, 3, 256, 256]))
        warmup = eff_cfg.get('warmup_runs', 3)
        runs = eff_cfg.get('timed_runs', 10)
        time_info = measure_inference_time(model, input_size, device, warmup, runs)
        print(f'  Inference time: {time_info["avg_ms"]:.1f} ± {time_info["std_ms"]:.1f} ms')

    # 4. 准备数据
    data_cfg = config['datasets']['val']
    lq_files = get_image_files(data_cfg['dataroot_lq'])
    gt_files = get_image_files(data_cfg['dataroot_gt'])
    mask_files = get_image_files(data_cfg['dataroot_mask'])

    assert len(lq_files) == len(gt_files), \
        f'LQ ({len(lq_files)}) and GT ({len(gt_files)}) count mismatch'
    if len(mask_files) == 1:
        mask_files = mask_files * len(lq_files)

    num_images = len(lq_files)
    print(f'\n[3/5] Processing {num_images} images...')

    # 可视化设置
    vis_cfg = config.get('visualize', {})
    vis_enabled = vis_cfg.get('enabled', False)
    vis_max = vis_cfg.get('max_images', 10)
    vis_dir = os.path.join(output_dir, 'visualizations')
    if vis_enabled:
        os.makedirs(vis_dir, exist_ok=True)

    # 评估设置
    eval_cfg = config.get('eval', {})
    eval_enabled = eval_cfg.get('enabled', False)

    # 5. 遍历推理
    all_metrics = []
    all_losses = []

    for idx in range(num_images):
        lq_path = lq_files[idx]
        gt_path = gt_files[idx]
        mask_path = mask_files[idx] if idx < len(mask_files) else mask_files[0]
        img_name = os.path.splitext(os.path.basename(lq_path))[0]

        # 加载数据
        lq = load_img(lq_path)
        gt = load_img(gt_path)
        mask = load_mask(mask_path)

        # Pad 到 multiple
        lq_pad, orig_h, orig_w = pad_to_multiple(lq, pad_multiple)
        gt_pad, _, _ = pad_to_multiple(gt, pad_multiple)
        mask_pad, _, _ = pad_to_multiple(mask, pad_multiple)

        # 推理
        output, intermediate = run_inference(model, lq_pad, mask_pad, device)

        # 裁剪回原始尺寸
        output = output[:, :, :orig_h, :orig_w]
        gt_crop = gt_pad[:, :, :orig_h, :orig_w].to(device)
        mask_crop = mask_pad[:, :, :orig_h, :orig_w].to(device)
        lq_crop = lq[:, :, :orig_h, :orig_w].to(device)

        # 保存输出图和 GT
        save_path = os.path.join(output_dir, 'images', f'{img_name}.png')
        save_img(output, save_path)
        gt_save_path = os.path.join(output_dir, 'images', f'{img_name}_gt.png')
        save_img(gt_crop, gt_save_path)

        # 可视化中间张量
        if vis_enabled and idx < vis_max:
            visualize_single(intermediate, vis_dir, img_name, gt=gt_crop)

        # 评估指标
        if eval_enabled:
            if eval_cfg.get('compute_psnr', False) or \
               eval_cfg.get('compute_deltae', False) or \
               eval_cfg.get('compute_ssim', False):
                metrics = compute_metrics(output, gt_crop, mask_crop)
                metrics['img_name'] = img_name
                all_metrics.append(metrics)

                # Debug: 第一张图打印详细信息
                if idx == 0:
                    out_np = output.squeeze(0).cpu().numpy()
                    gt_np = gt_crop.squeeze(0).cpu().numpy()
                    mse = np.mean((out_np - gt_np) ** 2)
                    print(f'  [DEBUG] {img_name}: output range=[{out_np.min():.4f}, {out_np.max():.4f}], '
                          f'gt range=[{gt_np.min():.4f}, {gt_np.max():.4f}], '
                          f'MSE={mse:.2e}, PSNR={metrics["psnr"]:.2f}')

            if eval_cfg.get('compute_losses', False) and intermediate is not None:
                losses = compute_all_losses(
                    output, gt_crop, mask_crop, lq_crop, intermediate, device
                )
                losses['img_name'] = img_name
                all_losses.append(losses)

        # 进度
        if (idx + 1) % 10 == 0 or idx == num_images - 1:
            print(f'  [{idx+1}/{num_images}] {img_name}')

    # 6. 汇总输出
    print(f'\n[4/5] Computing summary...')

    report = {
        'model': {
            'architecture': config['network_g']['type'],
            'weights_path': weights_path,
            'total_params': params_info['total'],
            'total_params_M': params_info['total_M'],
            'trainable_params': params_info['trainable'],
            'flops_G': flops_info['flops_G'],
        },
        'dataset': {
            'num_images': num_images,
            'input_dir': data_cfg['dataroot_lq'],
            'gt_dir': data_cfg['dataroot_gt'],
            'mask_dir': data_cfg['dataroot_mask'],
        },
        'metrics': {},
        'losses': {},
        'timing': time_info,
    }

    # 汇总 metrics
    if all_metrics:
        for key in ['psnr', 'deltaE', 'ssim']:
            values = [m[key] for m in all_metrics]
            report['metrics'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'per_image': [
                    {'img_name': m['img_name'], 'value': float(m[key])}
                    for m in all_metrics
                ],
            }

    # 汇总 losses
    if all_losses:
        loss_keys = [k for k in all_losses[0].keys() if k not in ('img_name', 'l_total')]
        for key in loss_keys:
            values = [l[key] for l in all_losses]
            report['losses'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'per_image': [
                    {'img_name': l['img_name'], 'value': float(l[key])}
                    for l in all_losses
                ],
            }

    # 保存 report
    report_path = os.path.join(output_dir, 'report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 7. 打印汇总
    print(f'\n[5/5] Results Summary')
    print('=' * 60)

    print(f'\nModel: {report["model"]["architecture"]}')
    print(f'Weights: {weights_path}')
    print(f'Params: {report["model"]["total_params_M"]:.2f}M')
    if report["model"]["flops_G"] > 0:
        print(f'FLOPs: {report["model"]["flops_G"]:.2f}G')
    if report["timing"]["avg_ms"] > 0:
        print(f'Inference: {report["timing"]["avg_ms"]:.1f} ± {report["timing"]["std_ms"]:.1f} ms')

    if report['metrics']:
        print(f'\nMetrics ({num_images} images):')
        for key, val in report['metrics'].items():
            print(f'  {key:12s}: {val["mean"]:.4f} ± {val["std"]:.4f}  '
                  f'(min={val["min"]:.4f}, max={val["max"]:.4f})')

    if report['losses']:
        print(f'\nLosses ({num_images} images):')
        for key, val in report['losses'].items():
            print(f'  {key:16s}: {val["mean"]:.6f} ± {val["std"]:.6f}')

    print(f'\nOutput saved to: {output_dir}')
    print(f'Report saved to: {report_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
