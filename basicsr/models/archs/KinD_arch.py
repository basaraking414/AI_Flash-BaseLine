"""
PyTorch re-implementation of KinD (DecomNet + Restoration + Illumination adjust)
Converted from the TensorFlow version present in the workspace. Exposes a
`KinDNet` class with a simple `forward(inp, mask=None)` API to match other archs.

Note: this is an architectural conversion to make the model usable inside the
project. Training losses/behaviour should follow the original implementation
when used in training scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def lrelu(x, negative_slope=0.2):
    return F.leaky_relu(x, negative_slope=negative_slope)


def conv_block(in_ch, out_ch, kernel_size=3, activation=True):
    pad = (kernel_size - 1) // 2
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=True)]
    if activation:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def upsample_and_concat(x1, x2, out_ch):
    # legacy helper kept for compatibility; prefer using per-module reductions
    x1_up = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=False)
    return torch.cat([x1_up, x2], dim=1)


class DecomNetSimple(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder
        self.conv1 = conv_block(3, 32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(64, 128)

        # decoder
        # project high-level features to desired channels before concat (mimic TF deconv)
        self.up3_reduce = nn.Conv2d(128, 64, kernel_size=1)
        self.conv8 = conv_block(64 + 64, 64)
        self.up8_reduce = nn.Conv2d(64, 32, kernel_size=1)
        self.conv9 = conv_block(32 + 32, 32)

        self.recon = nn.Conv2d(32, 3, kernel_size=1, padding=0)

        # illumination branch
        self.l_conv2 = conv_block(32, 32)
        self.l_conv4 = nn.Conv2d(32 + 32, 1, kernel_size=1, padding=0)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool(c1)
        c2 = self.conv2(p1)
        p2 = self.pool(c2)
        c3 = self.conv3(p2)

        up8 = F.interpolate(c3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        up8 = self.up3_reduce(up8)
        c8 = self.conv8(torch.cat([up8, c2], dim=1))

        up9 = F.interpolate(c8, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        up9 = self.up8_reduce(up9)
        c9 = self.conv9(torch.cat([up9, c1], dim=1))

        conv10 = self.recon(c9)
        R_out = torch.sigmoid(conv10)

        l_conv2 = self.l_conv2(c1)
        l_conv3 = torch.cat([l_conv2, c9], dim=1)
        L_out = torch.sigmoid(self.l_conv4(l_conv3))

        return R_out, L_out


class RestorationNet(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder
        self.conv1_1 = conv_block(6, 32)
        self.conv1_2 = conv_block(32, 32)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2_1 = conv_block(32, 64)
        self.conv2_2 = conv_block(64, 64)

        self.conv3_1 = conv_block(64, 128)
        self.conv3_2 = conv_block(128, 128)

        self.conv4_1 = conv_block(128, 256)
        self.conv4_2 = conv_block(256, 256)

        self.conv5_1 = conv_block(256, 512)
        self.conv5_2 = conv_block(512, 512)

        # decoder convs
        # reduce top-level features before concat to mimic conv2d_transpose in TF
        self.up5_reduce = nn.Conv2d(512, 256, kernel_size=1)
        self.conv6_1 = conv_block(256 + 256, 256)
        self.conv6_2 = conv_block(256, 256)

        self.up6_reduce = nn.Conv2d(256, 128, kernel_size=1)
        self.conv7_1 = conv_block(128 + 128, 128)
        self.conv7_2 = conv_block(128, 128)

        self.up7_reduce = nn.Conv2d(128, 64, kernel_size=1)
        self.conv8_1 = conv_block(64 + 64, 64)
        self.conv8_2 = conv_block(64, 64)

        self.up8_reduce = nn.Conv2d(64, 32, kernel_size=1)
        self.conv9_1 = conv_block(32 + 32, 32)
        self.conv9_2 = conv_block(32, 32)

        self.recon = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, input_r, input_i):
        # input_r/input_i assumed (B,3,H,W)
        x = torch.cat([input_r, input_i], dim=1)  # (B,6,H,W)

        c1 = self.conv1_1(x)
        c1 = self.conv1_2(c1)
        p1 = self.pool(c1)

        c2 = self.conv2_1(p1)
        c2 = self.conv2_2(c2)
        p2 = self.pool(c2)

        c3 = self.conv3_1(p2)
        c3 = self.conv3_2(c3)
        p3 = self.pool(c3)

        c4 = self.conv4_1(p3)
        c4 = self.conv4_2(c4)
        p4 = self.pool(c4)

        c5 = self.conv5_1(p4)
        c5 = self.conv5_2(c5)

        up6 = F.interpolate(c5, size=c4.shape[-2:], mode='bilinear', align_corners=False)
        up6 = self.up5_reduce(up6)
        c6 = self.conv6_1(torch.cat([up6, c4], dim=1))
        c6 = self.conv6_2(c6)

        up7 = F.interpolate(c6, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        up7 = self.up6_reduce(up7)
        c7 = self.conv7_1(torch.cat([up7, c3], dim=1))
        c7 = self.conv7_2(c7)

        up8 = F.interpolate(c7, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        up8 = self.up7_reduce(up8)
        c8 = self.conv8_1(torch.cat([up8, c2], dim=1))
        c8 = self.conv8_2(c8)

        up9 = F.interpolate(c8, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        up9 = self.up8_reduce(up9)
        c9 = self.conv9_1(torch.cat([up9, c1], dim=1))
        c9 = self.conv9_2(c9)

        out = torch.sigmoid(self.recon(c9))
        return out


class IlluminationAdjustNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(2, 32)
        self.conv2 = conv_block(32, 32)
        self.conv3 = conv_block(32, 32)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, input_i, input_ratio):
        x = torch.cat([input_i, input_ratio], dim=1)
        x = lrelu(self.conv1(x))
        x = lrelu(self.conv2(x))
        x = lrelu(self.conv3(x))
        out = torch.sigmoid(self.conv4(x))
        return out


class KinDNet(nn.Module):
    """Top-level KinD wrapper. forward(inp, mask=None) returns restored image.

    If you need the decomposition outputs, call `decompose` or use the submodules
    directly: `model.decom`, `model.restoration`, `model.illum_adjust`.
    """
    def __init__(self):
        super().__init__()
        self.decom = DecomNetSimple()
        self.restoration = RestorationNet()
        self.illum_adjust = IlluminationAdjustNet()

    def forward(self, inp, mask=None,alpha=None, ratio=None):
        # inp: (B,3,H,W)
        R, L = self.decom(inp)
        # Restoration_net expects illumination to be 3 channels (RGB-like). Expand L if needed.
        if L.shape[1] == 1:
            L3 = L.repeat(1, 3, 1, 1)
        else:
            L3 = L
        out = self.restoration(R, L3)
        return out

    def adjust_illumination(self, L, ratio):
        """Return adjusted illumination map given L and ratio map."""
        return self.illum_adjust(L, ratio)




