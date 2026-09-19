"""
RetinexMamba: Retinex-based Mamba for Low-Light Image Enhancement (CVPR 2024)

来源：https://github.com/YhuoyuH/RetinexMamba
适配 BasicSR 框架接口。

架构：
  Illumination_Estimator → illu_fea, illu_map
  input_img = img * illu_map + img
  Denoiser(input_img, illu_fea) → output

Denoiser 是 U-Net 结构：
  Embedding → Encoder(level=2) → Bottleneck → Decoder → Mapping + 残差
  每个 stage 使用 IGAB (IFA + SS2D(Mamba) + FFN)
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import _calculate_fan_in_and_fan_out

from basicsr.models.archs.RetinexMamba.SS2D_arch import SS2D
from basicsr.models.archs.RetinexMamba.IFA_arch import IFA


# ============================================================
# 权重初始化工具
# ============================================================

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_.")
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ============================================================
# 基础模块
# ============================================================

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


# ============================================================
# 光照估计器
# ============================================================

class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5,
                                     padding=2, bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)
        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


# ============================================================
# IGAB: Interleaved Group Attention Block (IFA + SS2D + FFN)
# ============================================================

class IGAB(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2, d_state=16):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IFA(dim_2=dim, dim=dim, num_heads=heads,
                    ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'),
                SS2D(d_model=dim, dropout=0, d_state=d_state),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        for (trans, ss2d, ff) in self.blocks:
            y = trans(x, illu_fea).permute(0, 2, 3, 1)
            x = ss2d(y) + x.permute(0, 2, 3, 1)
            x = ff(x) + x
            x = x.permute(0, 3, 1, 2)
        return x


# ============================================================
# Denoiser: U-Net encoder-decoder with Mamba blocks
# ============================================================

class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2,
                 num_blocks=[2, 4, 4], d_state=16):
        super().__init__()
        self.dim = dim
        self.level = level

        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i],
                     dim_head=dim, heads=dim_level // dim, d_state=d_state),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2
            d_state *= 2

        # Bottleneck
        self.bottleneck = IGAB(dim=dim_level, dim_head=dim,
                               heads=dim_level // dim,
                               num_blocks=num_blocks[-1], d_state=d_state)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2,
                                   stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i],
                     dim_head=dim, heads=(dim_level // 2) // dim, d_state=d_state)
            ]))
            dim_level //= 2
            d_state //= 2

        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        fea = self.embedding(x)
        fea_encoder = []
        illu_fea_list = []

        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        fea = self.bottleneck(fea, illu_fea)

        for i, (FeaUpSample, Fution, LeWinBlock) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlock(fea, illu_fea)

        out = self.mapping(fea) + x
        return out


# ============================================================
# RetinexMamba_Single_Stage
# ============================================================

class RetinexMamba_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31,
                 level=2, num_blocks=[1, 1, 1], d_state=16):
        super().__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels,
                                 dim=n_feat, level=level,
                                 num_blocks=num_blocks, d_state=d_state)

    def forward(self, img):
        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img, illu_fea)
        return output_img


# ============================================================
# RetinexMamba（原始多阶段模型）
# ============================================================

class RetinexMamba(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31,
                 stage=3, num_blocks=[1, 1, 1], d_state=16):
        super().__init__()
        self.stage = stage
        modules_body = [
            RetinexMamba_Single_Stage(
                in_channels=in_channels, out_dim=out_channels,
                n_feat=n_feat, level=2, num_blocks=num_blocks, d_state=d_state)
            for _ in range(stage)
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x)


# ============================================================
# RetinexMamba_AIFlash: 适配 BasicSR 接口（baseline）
# ============================================================

class RetinexMamba_AIFlash(nn.Module):
    """
    RetinexMamba baseline，适配 BasicSR 训练接口。
    mask 和 alpha 参数被忽略（RetinexMamba 不使用）。
    """

    def __init__(self, in_channels=3, out_channels=3, n_feat=31,
                 stage=3, num_blocks=[1, 1, 1], d_state=16):
        super().__init__()
        self.stage = stage
        modules_body = [
            RetinexMamba_Single_Stage(
                in_channels=in_channels, out_channels=out_channels,
                n_feat=n_feat, level=2, num_blocks=num_blocks, d_state=d_state)
            for _ in range(stage)
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, inp, mask=None, alpha=1.0):
        out = self.body(inp)
        return torch.clamp(out, 0, 1)


# ============================================================
# RetinexMamba_LIDM: RetinexMamba + DC-LIDM
# ============================================================

class RetinexMamba_LIDM(nn.Module):
    """
    RetinexMamba + DC-LIDM：在最后一个 stage 的 Denoiser bottleneck 处
    插入 DC-LIDM 模块进行 Retinex 分解。

    设计：
    - 前 stage-1 个 stage 使用原始 RetinexMamba_Single_Stage
    - 最后一个 stage 拆解为：Illumination_Estimator + Denoiser + DC-LIDM
    - sRGB→linear 转换在外部统一处理
    """

    def __init__(self, in_channels=3, out_channels=3, n_feat=32,
                 stage=3, num_blocks=[1, 1, 1], d_state=16, num_heads=4):
        super().__init__()
        self.stage = stage

        # 前 stage-1 个 stage 保持不变
        if stage > 1:
            self.pre_stages = nn.Sequential(*[
                RetinexMamba_Single_Stage(
                    in_channels=in_channels, out_channels=out_channels,
                    n_feat=n_feat, level=2, num_blocks=num_blocks, d_state=d_state)
                for _ in range(stage - 1)
            ])
        else:
            self.pre_stages = None

        # 最后一个 stage 的组件
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels,
                                 dim=n_feat, level=2,
                                 num_blocks=num_blocks, d_state=d_state)

        # DC-LIDM 插入在 Denoiser 的 bottleneck
        from basicsr.models.archs.DC_LIDM import DC_LIDM
        # Denoiser bottleneck 通道数 = n_feat * 2^level
        bottleneck_ch = n_feat * (2 ** 2)  # level=2
        self.dc_lidm = DC_LIDM(bottleneck_ch, num_heads=num_heads)
        self.n_feat = n_feat

    def forward(self, inp, mask=None, alpha=1.0):
        from basicsr.models.archs.DC_LIDM import srgb_to_linear, linear_to_srgb

        # 前 stage-1 个 stage（在 sRGB 空间）
        if self.pre_stages is not None:
            x = self.pre_stages(inp)
        else:
            x = inp

        # sRGB → linear
        x_linear = srgb_to_linear(x)

        # 最后一个 stage 的光照估计
        illu_fea, illu_map = self.estimator(x_linear)
        input_img = x_linear * illu_map + x_linear

        # Denoiser 前向，但需要获取 bottleneck 特征
        # 手动拆解 Denoiser.forward 以获取 bottleneck
        fea = self.denoiser.embedding(input_img)
        fea_encoder = []
        illu_fea_list = []

        for (IGAB_blk, FeaDownSample, IlluFeaDownsample) in self.denoiser.encoder_layers:
            fea = IGAB_blk(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        bottleneck = self.denoiser.bottleneck(fea, illu_fea)

        # Decoder
        fea = bottleneck
        for i, (FeaUpSample, Fution, LeWinBlock) in enumerate(self.denoiser.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.denoiser.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.denoiser.level - 1 - i]
            fea = LeWinBlock(fea, illu_fea)

        decoder_out = self.denoiser.mapping(fea) + input_img

        # DC-LIDM: Retinex 分解
        B, C, H, W = inp.shape
        result = self.dc_lidm(bottleneck, decoder_out[:, :, :H, :W],
                               mask, alpha, out_size=(H, W))

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
