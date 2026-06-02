"""
DC-LIDM: Dual-Constrained Latent Illumination Decomposition Module

独立的即插即用光照分解模块，可插入任意 encoder-decoder backbone。
从 Restormer_AIFlash_mask_attention 中提取并封装。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================
# sRGB / linear RGB 转换（标准 sRGB 传递函数）
# ============================================================

def srgb_to_linear(x):
    """sRGB → linear RGB"""
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4
    )


def linear_to_srgb(x):
    """linear RGB → sRGB"""
    return torch.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * torch.pow(x, 1 / 2.4) - 0.055
    )


class EnvLightHead(nn.Module):
    """环境光估计模块：Conv + Gaussian blur → 空间平滑的环境光图"""

    def __init__(self, in_dim, kernel_size=15, sigma=5.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_dim, 3, 1)
        )
        self.kernel_size = kernel_size
        self.register_buffer('gaussian_kernel',
                             self._make_gaussian_kernel(kernel_size, sigma))

    @staticmethod
    def _make_gaussian_kernel(k, sigma):
        ax = torch.arange(k, dtype=torch.float32) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        logits = self.conv(x)
        b, c, h, w = logits.shape
        kernel = self.gaussian_kernel.expand(c, 1, -1, -1)
        return F.conv2d(logits, kernel, padding=self.kernel_size // 2, groups=c)


class FlashHeadMasked(nn.Module):
    """闪光估计模块：Cross-Attention (mask→Q, latent→K,V) + Gated Fusion"""

    def __init__(self, latent_ch, out_ch=3, num_heads=4, use_specular=False, bias=False):
        super().__init__()
        assert latent_ch % num_heads == 0
        self.latent_ch = latent_ch
        self.num_heads = num_heads
        self.head_dim = latent_ch // num_heads
        self.scale = (self.head_dim) ** -0.5
        self.use_specular = use_specular

        self.q_conv = nn.Conv2d(1, latent_ch, kernel_size=1, bias=bias)
        self.k_conv = nn.Conv2d(latent_ch, latent_ch, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(latent_ch, latent_ch, kernel_size=1, bias=bias)
        self.ctx_out = nn.Conv2d(latent_ch, latent_ch, kernel_size=1, bias=bias)

        self.gate_conv = nn.Sequential(
            nn.Conv2d(latent_ch, latent_ch // 2, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_ch // 2, latent_ch, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )

        self.decode = nn.Sequential(
            nn.Conv2d(latent_ch, latent_ch, 3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_ch, out_ch if not use_specular else (3 + 1 + 1), 3, padding=1, bias=bias)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, latent, mask, mask_pool=None):
        B, C, Hl, Wl = latent.shape

        mask = mask[:, 0, :, :].unsqueeze(1)
        q_map = F.interpolate(mask.float(), size=(Hl, Wl), mode='bilinear', align_corners=False)
        if mask_pool is not None:
            q_map = mask_pool(q_map)

        q = self.q_conv(q_map)
        k = self.k_conv(latent)
        v = self.v_conv(latent)

        qt = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        kt = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        vt = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn = torch.einsum('b h c n, b h c m -> b h n m', qt, kt) * self.scale
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.einsum('b h n m, b h c m -> b h c n', attn, vt)

        ctx_sp = rearrange(ctx, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hl, w=Wl)
        ctx_proj = self.ctx_out(ctx_sp)
        if ctx_proj.shape[-2:] != (Hl, Wl):
            ctx_proj = F.interpolate(ctx_proj, size=(Hl, Wl), mode='bilinear', align_corners=False)

        g = self.gate_conv(ctx_proj)
        fused = latent * (1.0 - g) + ctx_proj * g
        out = self.decode(fused)

        if not self.use_specular:
            return out
        else:
            return out[:, 0:3, :, :], out[:, 3:4, :, :], out[:, 4:5, :, :]


class DC_LIDM(nn.Module):
    """
    Dual-Constrained Latent Illumination Decomposition Module

    接收 backbone 的 bottleneck 特征和 decoder 输出，
    输出 Retinex 分解的 R、E、F 以及合成结果。

    Args:
        bottleneck_ch: bottleneck 特征通道数
        num_heads: FlashHeadMasked 的注意力头数
        kernel_size: EnvLightHead 的 Gaussian 核大小
        sigma: EnvLightHead 的 Gaussian 核 sigma
    """

    def __init__(self, bottleneck_ch, num_heads=4, kernel_size=15, sigma=5.0):
        super().__init__()
        self.env_head = EnvLightHead(bottleneck_ch, kernel_size, sigma)
        self.flash_head = FlashHeadMasked(bottleneck_ch, out_ch=3, num_heads=num_heads)

    def forward(self, bottleneck_feat, decoder_out, mask, alpha=1.0, out_size=None):
        """
        Args:
            bottleneck_feat: encoder bottleneck 特征 (B, C, H', W')
            decoder_out: decoder 最终输出 (B, 3, H, W) — 用于提取 R
            mask: 人物 mask (B, 1, H, W)
            alpha: 闪光强度控制
            out_size: 输出空间尺寸，默认取 decoder_out 的尺寸

        Returns:
            dict: {
                'reflectance': R (B, 3, H, W), linear RGB, [0,1]
                'env_light': E (B, 3, H, W), linear RGB, [0,0.9]
                'flash_map': F (B, 3, H, W), linear RGB, [0,1]
                'illumination': E + alpha*F (B, 3, H, W), linear RGB
                'output': R * illumination (B, 3, H, W), linear RGB
                'alpha': float
            }
        """
        if out_size is None:
            out_size = decoder_out.shape[-2:]

        # R: reflectance from decoder
        reflectance = torch.sigmoid(decoder_out)

        # E: environmental light from bottleneck
        env_light_logits = self.env_head(bottleneck_feat)
        env_light = 0.9 * torch.sigmoid(env_light_logits)
        env_light_img = F.interpolate(env_light, size=out_size,
                                       mode='bilinear', align_corners=False)

        # F: flash from bottleneck + mask
        flash_logits = self.flash_head(bottleneck_feat, mask)
        flash_map = torch.sigmoid(F.interpolate(flash_logits, size=out_size,
                                                  mode='bilinear', align_corners=False))
        soft_mask = 0.9 * mask + 0.1
        flash_map = flash_map * soft_mask

        # 合成
        illumination = env_light_img + alpha * flash_map
        output = reflectance * illumination

        return {
            'reflectance': reflectance,
            'env_light': env_light_img,
            'flash_map': flash_map,
            'illumination': illumination,
            'alpha': alpha,
            'output': output,
        }
