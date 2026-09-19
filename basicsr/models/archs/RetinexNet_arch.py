"""
PyTorch implementation of RetinexNet (DecomNet + RelightNet) converted from the
TensorFlow definition used in the original RetinexNet repository.

This module exposes a `RetinexNet` nn.Module which performs decomposition into
reflectance and illumination and a relighting network to predict an illumination
adjustment. The forward API is compatible with basicsr-style arch modules:
    out = model(inp, mask)
where `mask` is optional and ignored by this simple implementation.

Notes:
- DecomNet: produces (R, I) where R is 3-channel reflectance (sigmoid) and I is
  1-channel illumination (sigmoid).
- RelightNet: predicts a 1-channel illumination delta map; the final enhanced
  image is R * I_delta (I_delta expanded to 3 channels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(in_ch, out_ch, kernel_size=3, stride=1, bias=True):
    pad = (kernel_size - 1) // 2
    return nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)


class DecomNet(nn.Module):
    """Decomposition network: input RGB image -> (R, I)

    Follows the TF implementation: prepend max-channel as 1-channel input and
    run several conv layers. Returns R (3ch sigmoid) and I (1ch sigmoid).
    """
    def __init__(self, layer_num=5, channel=64, kernel_size=3):
        super().__init__()
        # shallow feature extraction uses a larger kernel (kernel_size * 3) in TF
        ks0 = kernel_size * 3
        self.shallow = _conv(4, channel, kernel_size=ks0)
        body = []
        for _ in range(layer_num):
            body.append(_conv(channel, channel, kernel_size=kernel_size))
            body.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*body)
        self.recon = _conv(channel, 4, kernel_size=kernel_size)

    def forward(self, x):
        # x: (B,3,H,W)
        x_max, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)
        xin = torch.cat([x_max, x], dim=1)  # (B,4,H,W)

        out = self.shallow(xin)
        out = self.body(out)
        out = self.recon(out)

        R = torch.sigmoid(out[:, 0:3, :, :])
        I = torch.sigmoid(out[:, 3:4, :, :])
        return R, I


class RelightNet(nn.Module):
    """Relight network: takes illumination (1ch) and reflectance (3ch) -> 1ch delta

    Implements the encoder-decoder with nearest-neighbor upsampling and skip
    connections as in the TF code.
    """
    def __init__(self, channel=64, kernel_size=3):
        super().__init__()
        self.conv0 = _conv(4, channel, kernel_size=kernel_size)
        self.conv1 = _conv(channel, channel, kernel_size=kernel_size, stride=2)
        self.conv2 = _conv(channel, channel, kernel_size=kernel_size, stride=2)
        self.conv3 = _conv(channel, channel, kernel_size=kernel_size, stride=2)

        # decoder convs after upsampling
        self.dec_conv1 = _conv(channel, channel, kernel_size=kernel_size)
        self.dec_conv2 = _conv(channel, channel, kernel_size=kernel_size)
        self.dec_conv3 = _conv(channel, channel, kernel_size=kernel_size)

        # feature fusion and output
        self.fuse = _conv(channel * 3, channel, kernel_size=1)
        self.output = _conv(channel, 1, kernel_size=3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_I, input_R):
        # input_I: (B,1,H,W), input_R: (B,3,H,W)
        xin = torch.cat([input_R, input_I], dim=1)  # (B,4,H,W)
        conv0 = self.conv0(xin)
        conv1 = self.relu(self.conv1(conv0))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))

        # upsample conv3 -> size of conv2, conv1, conv0 sequentially
        up1 = F.interpolate(conv3, size=conv2.shape[-2:], mode='nearest')
        deconv1 = self.relu(self.dec_conv1(up1)) + conv2

        up2 = F.interpolate(deconv1, size=conv1.shape[-2:], mode='nearest')
        deconv2 = self.relu(self.dec_conv2(up2)) + conv1

        up3 = F.interpolate(deconv2, size=conv0.shape[-2:], mode='nearest')
        deconv3 = self.relu(self.dec_conv3(up3)) + conv0

        deconv1_resize = F.interpolate(deconv1, size=deconv3.shape[-2:], mode='nearest')
        deconv2_resize = F.interpolate(deconv2, size=deconv3.shape[-2:], mode='nearest')

        feature_gather = torch.cat([deconv1_resize, deconv2_resize, deconv3], dim=1)
        feature_fusion = self.fuse(feature_gather)
        out = self.output(feature_fusion)
        return out


class RetinexNet(nn.Module):
    """Top-level module exposing the RetinexNet pipeline.

    forward(inp, mask=None) -> enhanced image (B,3,H,W)
    """
    def __init__(self, decom_layer_num=5, channel=64, kernel_size=3):
        super().__init__()
        self.decom = DecomNet(layer_num=decom_layer_num, channel=channel, kernel_size=kernel_size)
        self.relight = RelightNet(channel=channel, kernel_size=kernel_size)

        # small weight init for convs
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, inp, mask=None,alpha=None):
        """Compute enhanced output from input image.

        Args:
            inp: (B,3,H,W) tensor in range [0,1]
            mask: optional, ignored (kept for API compatibility)

        Returns:
            out_img: (B,3,H,W) enhanced image tensor
        """
        R, I = self.decom(inp)
        I_delta = self.relight(I, R)  # (B,1,H,W)

        # expand I_delta to 3 channels to multiply with reflectance
        I_delta_3 = I_delta.repeat(1, 3, 1, 1)
        out = R * I_delta_3

        out = torch.sigmoid(out)

        # vis = torch.cat([inp,R,I_delta_3,out], dim=-1)
        # visualize_tensor(vis)
        return out

import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision import transforms
def visualize_tensor(tensor, title=None, save_path=None):
    """
    将PyTorch Tensor转换为图像并显示

    参数:
    - tensor (torch.Tensor): 需要可视化的Tensor，支持以下格式:
        * (C, H, W) - 无批量维度的单张图像
        * (B, C, H, W) - 带批量维度的多张图像
        * (H, W, C) - 通道在最后的格式
    - title (str, 可选): 图像标题
    - save_path (str, 可选): 保存图像的路径，若为None则不保存

    返回:
    - None
    """
    # 确保Tensor在CPU上
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # 克隆Tensor避免修改原始数据
    tensor = tensor.detach().clone()

    # 处理批量维度: 选择第一张图像或移除单例批量维度
    if tensor.dim() == 4:
        tensor = tensor[0]  # 选择批量中的第一张图像

    # 确保维度顺序为(C, H, W)
    if tensor.shape[0] in [1, 3]:  # 如果第一个维度是通道数
        tensor = tensor.permute(1, 2, 0)  # 从(H, W, C)转为(C, H, W)

    # 处理单通道情况
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)  # 移除通道维度
        is_grayscale = True
    else:
        is_grayscale = False

    # 转换为numpy数组并调整值范围
    tensor_np = tensor.numpy()

    # 检查值范围并调整到[0, 1]或[0, 255]
    if tensor_np.max() > 1.0:
        tensor_np = np.clip(tensor_np,0,1)
    elif tensor_np.min() < 0:
        tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min())

    # 显示图像
    plt.ion()
    plt.figure(figsize=(10, 8))
    if is_grayscale:
        plt.imshow(tensor_np, cmap='gray')
    else:
        plt.imshow(tensor_np)
    plt.draw()  # 强制刷新
    plt.pause(0.1)
    plt.axis('off')

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()

    # 保存图像（如果指定路径）
    if save_path:
        if is_grayscale:
            plt.imsave(save_path, tensor_np, cmap='gray')
        else:
            plt.imsave(save_path, tensor_np)