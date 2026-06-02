"""
SCI: Self-Calibrated Illumination (CVPR 2022)

来源：E:\ricky\research\AI Flash Potrait\SCI-main\SCI-main\CVPR\model.py
适配 BasicSR 框架接口。
"""

import torch
import torch.nn as nn


class EnhanceNetwork(nn.Module):
    """光照估计网络：估计输入图像的光照图"""

    def __init__(self, layers, channels):
        super().__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size, 1, padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        fea = self.in_conv(x)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        illu = fea + x
        illu = torch.clamp(illu, 0.0001, 1)
        return illu


class CalibrateNetwork(nn.Module):
    """校正网络：对反射率进行校正"""

    def __init__(self, layers, channels):
        super().__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size, 1, padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        fea = self.in_conv(x)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        delta = x - fea
        return delta


class SCI_Net(nn.Module):
    """
    SCI 包装类，适配 BasicSR 接口。

    原始 Network.forward() 返回 4 个列表，本类取最后一个 stage 的
    反射率 rlist[-1] 作为最终输出。

    Args:
        stage: 迭代精炼次数（默认 3）
    """

    def __init__(self, stage=3):
        super().__init__()
        self.stage = stage
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self.calibrate = CalibrateNetwork(layers=3, channels=16)

    def forward(self, inp, mask=None, alpha=1.0):
        """
        Args:
            inp: (B, 3, H, W) 输入图像，[0, 1]
            mask: 忽略（SCI 不使用 mask）
            alpha: 忽略（SCI 不支持 alpha 控制）

        Returns:
            out: (B, 3, H, W) 输出图像，[0, 1]
        """
        input_op = inp
        for _ in range(self.stage):
            i = self.enhance(input_op)
            r = inp / i
            r = torch.clamp(r, 0, 1)
            att = self.calibrate(r)
            input_op = inp + att

        return torch.clamp(input_op, 0, 1)


class SCI_Net_LIDM(nn.Module):
    """SCI + DC-LIDM：带 Retinex 光照分解的 SCI"""

    def __init__(self, stage=3, bottleneck_ch=16, num_heads=4):
        super().__init__()
        self.stage = stage
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self.calibrate = CalibrateNetwork(layers=3, channels=16)
        from basicsr.models.archs.dc_lidm import DC_LIDM
        self.dc_lidm = DC_LIDM(bottleneck_ch, num_heads=num_heads)
        self.bottleneck_ch = bottleneck_ch

    def forward(self, inp, mask=None, alpha=1.0):
        from basicsr.models.archs.dc_lidm import srgb_to_linear, linear_to_srgb

        inp_linear = srgb_to_linear(inp)
        input_op = inp_linear

        for _ in range(self.stage):
            i = self.enhance(input_op)
            r = inp_linear / i
            r = torch.clamp(r, 0, 1)
            att = self.calibrate(r)
            input_op = inp_linear + att

        # 用 enhance 的输出作为 bottleneck 特征（扩展通道数）
        bottleneck = i.repeat(1, self.bottleneck_ch // 3, 1, 1)[:, :self.bottleneck_ch]
        decoder_out = input_op

        result = self.dc_lidm(bottleneck, decoder_out, mask, alpha)

        out_img = linear_to_srgb(result['output'])
        out_img = torch.clamp(out_img, 0, 1)

        self._intermediate = {
            'reflectance': result['reflectance'],
            'env_light': result['env_light'],
            'flash_map': result['flash_map'],
            'alpha': result['alpha'],
            'illumination': result['illumination'],
        }
        return out_img
