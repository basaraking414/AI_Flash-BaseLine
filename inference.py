"""
AI Flash Portrait Inference Script

使用示例:
    # 基础推理（无评估）
    python inference.py --config configs/my_aiflash.yaml --weights path/to/model.pth --input_dir ./test/input --mask_dir ./test/mask --output_dir ./results

    # 带评估指标（需要 GT）
    python inference.py --config configs/my_aiflash.yaml --weights path/to/model.pth --input_dir ./test/input --mask_dir ./test/mask --gt_dir ./test/gt --output_dir ./results --eval

    # 使用 tile 模式处理大图
    python inference.py --config configs/my_aiflash.yaml --weights path/to/model.pth --input_dir ./test/input --mask_dir ./test/mask --output_dir ./results --tile 512

    # 指定模型架构
    python inference.py --config configs/my_aiflash_ablation.yaml --weights path/to/model.pth --input_dir ./test/input --mask_dir ./test/mask --arch Restormer_AIFlash_mask_attention
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from skimage import img_as_ubyte


def parse_args():
    parser = argparse.ArgumentParser(description='AI Flash Portrait Inference')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights (.pth file)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory or single image path')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Mask directory or single mask path')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Ground truth directory (required for --eval)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for restored images')
    parser.add_argument('--arch', type=str, default='Restormer_AIFlash',
                        choices=['Restormer_AIFlash', 'Restormer_AIFlash_mask_attention'],
                        help='Model architecture to use')
    parser.add_argument('--eval', action='store_true',
                        help='Enable evaluation mode (compute PSNR and DeltaE)')
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size for processing large images')
    parser.add_argument('--tile_overlap', type=int, default=32,
                        help='Overlap between tiles')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(config, arch_name):
    if arch_name == 'Restormer_AIFlash_mask_attention':
        from basicsr.models.archs.my_restormer_arch import Restormer_AIFlash_mask_attention as ArchClass
    else:
        from basicsr.models.archs.my_restormer_arch import Restormer_AIFlash as ArchClass

    network_config = config['network_g']
    model_params = {k: v for k, v in network_config.items() if k != 'type'}
    return ArchClass(**model_params)


def load_weights(model, weights_path):
    print(f'Loading weights from: {weights_path}')
    checkpoint = torch.load(weights_path, map_location='cpu')

    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v

    model.load_state_dict(new_state_dict, strict=True)
    print('Weights loaded successfully!')
    return model


def load_img(filepath):
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f'Failed to load image: {filepath}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def get_image_files(input_path):
    extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'bmp', 'BMP']

    if os.path.isfile(input_path):
        return [input_path]

    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(input_path, f'*.{ext}')))

    return natsorted(files)


def pad_to_multiple(img_tensor, multiple=8):
    _, _, h, w = img_tensor.shape
    H = ((h + multiple) // multiple) * multiple
    W = ((w + multiple) // multiple) * multiple
    padh = H - h if h % multiple != 0 else 0
    padw = W - w if w % multiple != 0 else 0

    if padh > 0 or padw > 0:
        img_tensor = F.pad(img_tensor, (0, padw, 0, padh), mode='reflect')

    return img_tensor, h, w


def inference_full(model, input_tensor, mask_tensor):
    return model(input_tensor, mask_tensor)


def inference_tile(model, input_tensor, mask_tensor, tile_size, tile_overlap):
    b, c, h, w = input_tensor.shape
    tile = min(tile_size, h, w)

    assert tile % 8 == 0, "Tile size should be multiple of 8"

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

    E = torch.zeros(b, c, h, w).type_as(input_tensor)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = input_tensor[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            mask_patch = mask_tensor[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = model(in_patch, mask_patch)

            if isinstance(out_patch, list):
                out_patch = out_patch[-1]

            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(torch.ones_like(out_patch))

    return E.div_(W)


@torch.no_grad()
def process_image(model, img_path, mask_path, output_dir, device, gt_path=None, tile_size=None, tile_overlap=32, eval_mode=False):
    img = load_img(img_path)
    mask = load_img(mask_path) if mask_path is not None else None

    input_tensor = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device) if mask is not None else torch.ones_like(input_tensor)

    input_padded, padded_h, padded_w = pad_to_multiple(input_tensor, multiple=8)
    mask_padded, _, _ = pad_to_multiple(mask_tensor, multiple=8)

    if tile_size is None:
        output = inference_full(model, input_padded, mask_padded)
    else:
        output = inference_tile(model, input_padded, mask_padded, tile_size, tile_overlap)

    if isinstance(output, list):
        output = output[-1]

    output = torch.clamp(output, 0, 1)
    output = output[:, :, :padded_h, :padded_w]

    output_np = output.permute(0, 2, 3, 1).cpu().detach().numpy()
    output_np = img_as_ubyte(output_np[0])

    filename = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_dir, f'{filename}.png')
    save_img(output_path, output_np)

    psnr, deltaE = None, None
    if eval_mode and gt_path is not None:
        from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_deltaE
        gt = load_img(gt_path)
        gt_tensor = torch.from_numpy(gt).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
        gt_padded, _, _ = pad_to_multiple(gt_tensor, multiple=8)
        psnr = calculate_psnr(output, gt_padded, mask_padded)
        deltaE = calculate_deltaE(output, gt_padded, mask_padded)

    return output_path, psnr, deltaE


def main():
    args = parse_args()

    if args.eval and args.gt_dir is None:
        raise ValueError("--gt_dir is required when --eval is enabled")

    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    config = load_config(args.config)
    model = build_model(config, args.arch)
    model = load_weights(model, args.weights)
    model = model.to(device).eval()

    network_config = config['network_g']
    print(f"\nModel Config:")
    print(f"  - Arch: {args.arch}")
    print(f"  - Type: {network_config.get('type', 'Restormer')}")
    print(f"  - Dim: {network_config.get('dim', 48)}")

    image_files = get_image_files(args.input_dir)
    mask_files = get_image_files(args.mask_dir)
    gt_files = get_image_files(args.gt_dir) if args.gt_dir else None

    if len(image_files) == 0:
        raise ValueError(f'No image files found in: {args.input_dir}')

    if args.max_images:
        image_files = image_files[:args.max_images]

    print(f'\nFound {len(image_files)} images to process')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.tile:
        print(f'Using tile mode: tile_size={args.tile}, overlap={args.tile_overlap}')
    else:
        print('Using full resolution mode')

    psnr_list, deltaE_list = [], []

    for i in tqdm(range(len(image_files)), desc='Processing'):
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        img_path = image_files[i]
        mask_path = mask_files[i] if i < len(mask_files) else None
        gt_path = gt_files[i] if gt_files and i < len(gt_files) else None

        try:
            output_path, psnr, deltaE = process_image(
                model=model,
                img_path=img_path,
                mask_path=mask_path,
                output_dir=args.output_dir,
                device=device,
                gt_path=gt_path,
                tile_size=args.tile,
                tile_overlap=args.tile_overlap,
                eval_mode=args.eval
            )
            if psnr is not None:
                psnr_list.append(psnr)
            if deltaE is not None:
                deltaE_list.append(deltaE)
        except Exception as e:
            print(f'\nError processing {img_path}: {e}')
            continue

    if args.eval and psnr_list:
        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_deltaE = sum(deltaE_list) / len(deltaE_list)
        print(f'\nEvaluation Results:')
        print(f'  - Average PSNR: {avg_psnr:.2f} dB')
        print(f'  - Average DeltaE: {avg_deltaE:.2f}')

    print(f'\nInference completed! Results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
