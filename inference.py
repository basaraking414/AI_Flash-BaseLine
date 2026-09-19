"""
AI-Flash / Restormer 推理脚本
从 YAML 配置文件加载网络结构，加载 best_model 中的权重进行推理。

使用示例:
    # 默认用法：configs/my_aiflash.yaml + best_model/best_Restormer_LIDM.pth
    #             输入 dataset/test/input/ + dataset/test/mask_personmask/
    #             输出 result/test/
    python inference.py

    # 显式指定配置与权重
    python inference.py --config configs/my_aiflash.yaml \
                        --weights best_model/best_Restormer_LIDM.pth

    # 指定输入 / 掩码 / 输出目录
    python inference.py --input_dir dataset/test/input \
                        --mask_dir dataset/test/mask_personmask \
                        --output_dir result/test

    # 显存不足时使用 tile 模式处理大图
    python inference.py --tile 512 --tile_overlap 32

    # 处理单张图片
    python inference.py --input_dir dataset/test/input/001.jpg \
                        --mask_dir dataset/test/mask_personmask/001.jpg

    # 强制使用 CPU
    python inference.py --device cpu
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from natsort import natsorted
from skimage import img_as_ubyte


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def parse_args():
    parser = argparse.ArgumentParser(description='AI-Flash / Restormer Inference')
    parser.add_argument('--config', type=str, default='configs/my_aiflash.yaml',
                        help='Path to YAML config file (network_g section is used)')
    parser.add_argument('--weights', type=str, default='best_model/best_Restormer_LIDM.pth',
                        help='Path to model weights (.pth file)')
    parser.add_argument('--input_dir', type=str, default='dataset/test/input',
                        help='Input directory or single image path')
    parser.add_argument('--mask_dir', type=str, default='dataset/test/mask_personmask',
                        help='Mask directory or single mask path')
    parser.add_argument('--output_dir', type=str, default='result/test',
                        help='Output directory for restored images')
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size for processing large images (e.g. 512, 720). None means full resolution.')
    parser.add_argument('--tile_overlap', type=int, default=32,
                        help='Overlap between tiles')
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use: 'auto', 'cuda', 'cuda:0' or 'cpu'")
    return parser.parse_args()


def resolve_device(device_arg):
    """解析设备参数，cuda 不可用时回退到 cpu"""
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_arg.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA not available, falling back to CPU')
        return torch.device('cpu')
    return torch.device(device_arg)


def load_config(config_path):
    """加载 YAML 配置文件"""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if 'network_g' not in config:
        raise KeyError(f"'network_g' section not found in {config_path}")
    return config


def build_model(config):
    """根据配置构建模型"""
    from basicsr.models.archs.my_restormer_arch import Restormer_AIFlash_mask_attention

    # 从配置文件获取网络参数
    network_config = config['network_g']

    # 移除 'type' 字段，只保留模型参数
    model_params = {k: v for k, v in network_config.items() if k != 'type'}

    # 构建模型
    model = Restormer_AIFlash_mask_attention(**model_params)

    return model


def load_weights(model, weights_path):
    """加载模型权重"""
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f'Weights not found: {weights_path}\n'
            f'  权重文件不随仓库分发，请参考 README.md 的 "权重与数据" 一节放置文件。')

    print(f'Loading weights from: {weights_path}')
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
    except Exception:
        # 旧格式 / 非纯 tensor 的 checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

    # 处理不同格式的权重文件
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and len(checkpoint) == 1 and \
            isinstance(next(iter(checkpoint.values())), dict):
        # 形如 {'net_g_latest': {...}} 的历史存档，自动解包
        state_dict = next(iter(checkpoint.values()))
    else:
        state_dict = checkpoint

    # 移除可能的 'module.' 前缀（多卡训练产生的）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    print('Weights loaded successfully!')
    return model


def load_img(filepath):
    """加载图像 (BGR -> RGB, uint8)"""
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Failed to load image: {filepath}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(filepath):
    """
    加载人像掩码，返回单通道 float32 数组（取值 {0, 1}）

    与训练侧保持一致：数据集以灰度方式读取掩码后做 mask > 0 的二值化。
    """
    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f'Failed to load mask: {filepath}')
    mask = mask.astype(np.float32) / 255.
    return (mask > 0).astype(np.float32)


def save_img(filepath, img):
    """保存图像 (RGB -> BGR)"""
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def get_image_files(input_path):
    """
    获取输入图像文件列表

    目录模式下按扩展名筛选；扩展名统一转小写后比较，
    避免 Windows 上 '*.jpg' 与 '*.JPG' 命中同一批文件导致重复处理。
    """
    # 判断是单个文件还是目录
    if os.path.isfile(input_path):
        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f'Input path not found: {input_path}')

    # 目录模式：搜索所有图像文件
    files = [
        os.path.join(input_path, name)
        for name in os.listdir(input_path)
        if os.path.splitext(name)[1].lower() in IMG_EXTENSIONS
        and os.path.isfile(os.path.join(input_path, name))
    ]

    return natsorted(files)


def build_mask_lookup(mask_path):
    """
    建立 文件名(不含扩展名) -> 掩码路径 的映射。
    这样即使掩码与图像数量不一致、或排序不同，也能按名字配对。
    """
    lookup = {}
    for path in get_image_files(mask_path) if mask_path else []:
        key = os.path.splitext(os.path.basename(path))[0]
        lookup.setdefault(key, path)
    return lookup


def find_mask_for(img_path, mask_lookup, mask_list, index):
    """按文件名配对掩码；失败时退回按顺序配对"""
    key = os.path.splitext(os.path.basename(img_path))[0]
    if key in mask_lookup:
        return mask_lookup[key]
    if index < len(mask_list):
        print(f'  warning: no mask named "{key}", fallback to the {index}-th mask')
        return mask_list[index]
    return None


def pad_to_multiple(img_tensor, multiple=8):
    """
    将图像 pad 到 multiple 的倍数
    返回: (padded_tensor, original_h, original_w)
    """
    _, _, h, w = img_tensor.shape
    H = ((h + multiple) // multiple) * multiple
    W = ((w + multiple) // multiple) * multiple
    padh = H - h if h % multiple != 0 else 0
    padw = W - w if w % multiple != 0 else 0

    if padh > 0 or padw > 0:
        img_tensor = F.pad(img_tensor, (0, padw, 0, padh), mode='reflect')

    return img_tensor, h, w


def inference_full(model, input_tensor, mask_tensor):
    """全图推理"""
    return model(input_tensor, mask_tensor)


def inference_tile(model, input_tensor, mask_tensor, tile_size, tile_overlap):
    """
    分块推理（用于处理大图，显存不足时使用）
    """
    b, c, h, w = input_tensor.shape
    tile = min(tile_size, h, w)

    assert tile % 8 == 0, "Tile size should be multiple of 8"

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

    # 输出和权重累加器
    E = torch.zeros(b, c, h, w).type_as(input_tensor)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = input_tensor[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            mask_patch = mask_tensor[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            out_patch = model(in_patch, mask_patch)

            # 处理可能的 list 输出
            if isinstance(out_patch, list):
                out_patch = out_patch[-1]

            E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
            W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(1)

    # 平均重叠区域
    output = E.div_(W)

    return output


@torch.no_grad()
def process_image(model, img_path, mask_path, output_dir, device,
                  tile_size=None, tile_overlap=32):
    """处理单张图像"""
    img = load_img(img_path)
    h, w = img.shape[:2]

    if mask_path is None:
        print(f'  warning: no mask for {os.path.basename(img_path)}, use an all-zero mask')
        mask = np.zeros((h, w), dtype=np.float32)
    else:
        mask = load_mask(mask_path)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)

    # 转换为 tensor: (H, W, C) -> (1, 3, H, W)；mask -> (1, 1, H, W)，归一化到 [0, 1]
    input_tensor = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)

    # Pad 到 8 的倍数
    input_padded, padded_h, padded_w = pad_to_multiple(input_tensor, multiple=8)
    mask_padded, _, _ = pad_to_multiple(mask_tensor, multiple=8)

    if tile_size is None:
        output = inference_full(model, input_padded, mask_padded)
    else:
        output = inference_tile(model, input_padded, mask_padded, tile_size, tile_overlap)

    # 处理可能的 list 输出
    if isinstance(output, list):
        output = output[-1]

    output = torch.clamp(output, 0, 1)
    output = output[:, :, :padded_h, :padded_w]

    # 转换回 numpy: (1, C, H, W) -> (H, W, C)
    output_np = output.permute(0, 2, 3, 1).cpu().detach().numpy()
    output_np = img_as_ubyte(output_np[0])

    filename = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_dir, f'{filename}.png')
    save_img(output_path, output_np)

    return output_path


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f'Using device: {device}')

    # 加载配置
    print(f'Loading config from: {args.config}')
    config = load_config(args.config)

    # 构建模型
    print('Building model...')
    model = build_model(config)
    model = load_weights(model, args.weights)
    model = model.to(device)
    model.eval()

    # 打印模型信息
    network_config = config['network_g']
    print("\nModel Config:")
    print(f"  - Type: {network_config.get('type', 'Restormer')}")
    print(f"  - Dim: {network_config.get('dim', 48)}")
    print(f"  - Num Blocks: {network_config.get('num_blocks', [4, 6, 6, 8])}")
    print(f"  - Heads: {network_config.get('heads', [1, 2, 4, 8])}")

    # 获取输入文件
    image_files = get_image_files(args.input_dir)
    if len(image_files) == 0:
        raise ValueError(f'No image files found in: {args.input_dir}')

    mask_lookup = build_mask_lookup(args.mask_dir)
    mask_list = natsorted(mask_lookup.values()) if mask_lookup else []
    if len(mask_lookup) == 0:
        print(f'warning: no mask files found in: {args.mask_dir}')

    print(f'\nFound {len(image_files)} images to process, {len(mask_list)} masks available')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output directory: {args.output_dir}')

    # 推理设置
    if args.tile:
        print(f'Using tile mode: tile_size={args.tile}, overlap={args.tile_overlap}')
    else:
        print('Using full resolution mode')

    print('\n' + '=' * 50)
    print('Starting inference...')
    print('=' * 50 + '\n')

    # 处理所有图像
    done, failed = 0, 0
    for i, img_path in enumerate(image_files):
        mask_path = find_mask_for(img_path, mask_lookup, mask_list, i)

        # 清理 GPU 缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        try:
            output_path = process_image(
                model=model,
                img_path=img_path,
                mask_path=mask_path,
                output_dir=args.output_dir,
                device=device,
                tile_size=args.tile,
                tile_overlap=args.tile_overlap
            )
            done += 1
            print(f'[{done}/{len(image_files)}] {os.path.basename(img_path)} -> {output_path}')
        except Exception as e:
            failed += 1
            print(f'\nError processing {img_path}: {e}')
            continue

    print(f'\n{"=" * 50}')
    print(f'Inference completed! {done} succeeded, {failed} failed.')
    print(f'Results saved to: {args.output_dir}')
    print(f'{"=" * 50}')

    if done == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()