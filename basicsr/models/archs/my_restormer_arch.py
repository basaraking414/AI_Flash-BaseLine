## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class EnvLightHead(nn.Module):   #增加环境背景光源估计模块
    def __init__(self, in_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.ReLU(),
            nn.Conv2d(in_dim, 3, 1)   # RGB illumination
        )

    def forward(self, x):
        x = self.pool(x)
        env_light = self.fc(x)
        return env_light

class FlashLightHead(nn.Module):   #增加闪光灯光照强度估计模块
    def __init__(self, in_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_dim, 3, 1)  # rgb map
        )

    def forward(self, x):
        flash_map = self.conv(x)
        return flash_map
##########################################################################
##---------- Restormer -----------------------
class Restormer_AIFlash(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer_AIFlash, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.env_head = EnvLightHead(int(dim * 2 ** 3))    #添加模块
        self.flash_head = FlashLightHead(int(dim * 2 ** 3))

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, mask):
        inp_img = srgb_to_linear(inp_img)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4)

        env_light = self.env_head(latent)  # (B,3,1,1)  估计环境光
        flash_map = self.flash_head(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)


        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1)

        reflectance = torch.sigmoid(out_dec_level1)

        env_light = 0.9 * torch.sigmoid(env_light)
        env_light_img = env_light.expand_as(reflectance)

        flash_map = 2 * torch.sigmoid(flash_map)
        flash_map = F.interpolate(flash_map, size=reflectance.shape[-2:])

        soft_mask = 0.9 * mask + 0.1
        # radial = compute_gaussian_radial_map(mask)#根据mask计算径向衰减图
        flash_map = flash_map * soft_mask  #增加软掩膜和径向衰减，模拟闪光灯光照强度随距离衰减的效果

        illumination = env_light_img + flash_map

        out_img = reflectance * illumination

        # out_img = torch.clamp(out_img, 0, 1)
        out_img = linear_to_srgb(out_img)

        # visualize_tensor(env_light_img)
        # visualize_tensor(flash_map)
        # visualize_tensor(illumination)
        # visualize_tensor(out_img)
        return out_img


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class FlashHeadMasked(nn.Module):
    """
    Flash head that takes (latent, mask) -> outputs flash_map (B x 3 x h_out x w_out).
    Options:
      - use_specular: if True, output (diffuse_3ch, spec_mask_1ch, spec_strength_1ch)
      - num_heads: cross-attention heads for mask->latent
    Design:
      1) project mask -> queries (q)
      2) project latent -> keys/values (k,v)
      3) compute cross-attention, produce context
      4) fuse context with latent (residual + gating)
      5) decode fused latent to flash_map (and optional spec parts)
    """
    def __init__(self, latent_ch, out_ch=3, num_heads=4, use_specular=False, bias=False):
        super().__init__()
        assert latent_ch % num_heads == 0
        self.latent_ch = latent_ch
        self.num_heads = num_heads
        self.head_dim = latent_ch // num_heads
        self.scale = (self.head_dim) ** -0.5
        self.use_specular = use_specular

        # mask -> q projection
        self.q_conv = nn.Conv2d(1, latent_ch, kernel_size=1, bias=bias)

        # latent -> k, v projections
        self.k_conv = nn.Conv2d(latent_ch, latent_ch, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(latent_ch, latent_ch, kernel_size=1, bias=bias)

        # output projection of attention context
        self.ctx_out = nn.Conv2d(latent_ch, latent_ch, kernel_size=1, bias=bias)

        # gating fusion: learnable conv to produce gating map in (0,1)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(latent_ch, latent_ch//2, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_ch//2, latent_ch, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )

        # decoder to produce flash prediction at latent resolution
        self.decode = nn.Sequential(
            nn.Conv2d(latent_ch, latent_ch, 3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_ch, out_ch if not use_specular else (3 + 1 + 1), 3, padding=1, bias=bias)
        )

        # small init for stability
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, latent, mask, mask_pool=None):
        """
        latent: B x C x Hl x Wl
        mask:   B x 1 x H x W (original image resolution)
        mask_pool: optional nn.Module to downsample mask tokens (e.g., AvgPool2d)
        returns:
           if not use_specular: flash_map (B x 3 x Hl x Wl)
           if use_specular: tuple (diffuse, spec_mask, spec_strength)
        """
        B, C, Hl, Wl = latent.shape

        mask = mask[:,0,:,:]
        mask = mask.unsqueeze(1)
        # 1) resize mask to latent resolution
        q_map = F.interpolate(mask.float(), size=(Hl, Wl), mode='bilinear', align_corners=False)
        if mask_pool is not None:
            q_map = mask_pool(q_map)  # optional further reduction

        # 2) projections
        q = self.q_conv(q_map)    # B x C x Hq x Wq
        k = self.k_conv(latent)   # B x C x Hl x Wl
        v = self.v_conv(latent)   # B x C x Hl x Wl

        # reshape for multihead: B x heads x head_dim x N
        qt = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        kt = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        vt = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 3) attention: q @ k^T -> B x heads x Nq x Nk
        attn = torch.einsum('b h c n, b h c m -> b h n m', qt, kt) * self.scale
        attn = torch.softmax(attn, dim=-1)

        # 4) context = attn @ v -> b h c nq
        ctx = torch.einsum('b h n m, b h c m -> b h c n', attn, vt)

        # 5) reshape context back to spatial (Hq x Wq)
        ctx_sp = rearrange(ctx, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hl, w=Wl)

        # 6) project context and upsample if needed
        ctx_proj = self.ctx_out(ctx_sp)  # B x C x Hq x Wq
        if ctx_proj.shape[-2:] != (Hl, Wl):
            ctx_proj = F.interpolate(ctx_proj, size=(Hl, Wl), mode='bilinear', align_corners=False)

        # 7) gated fusion: latent <- latent * (1 - g) + ctx_proj * g
        g = self.gate_conv(ctx_proj)  # B x C x Hl x Wl (0..1)
        fused = latent * (1.0 - g) + ctx_proj * g

        # 8) decode to flash output at latent resolution
        out = self.decode(fused)  # B x out_ch_or_mixed x Hl x Wl

        if not self.use_specular:
            # out: B x 3 x Hl x Wl (raw logits), caller should upsample & sigmoid/relu as needed
            return out
        else:
            # split: diffuse(3), spec_mask(1), spec_strength(1)
            diffuse = out[:, 0:3, :, :]
            spec_mask = out[:, 3:4, :, :]
            spec_strength = out[:, 4:5, :, :]
            return diffuse, spec_mask, spec_strength

class Restormer_AIFlash_mask_attention(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer_AIFlash_mask_attention, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.env_head = EnvLightHead(int(dim * 2 ** 3))  # 添加模块
        self.flash_head = FlashHeadMasked(latent_ch=int(dim * 2 ** 3), out_ch=3, num_heads=4, use_specular=False)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, mask):
        s=inp_img
        # inp_img = srgb_to_linear(inp_img)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1)

        reflectance = torch.sigmoid(out_dec_level1)

        env_light = self.env_head(latent)  # (B,3,1,1)  估计环境光

        flash_logits = self.flash_head(latent, mask, getattr(self, 'mask_pool', None))
        flash_map = torch.sigmoid(F.interpolate(flash_logits, size=reflectance.shape[-2:], mode='bilinear', align_corners=False))

        env_light = 0.9 * torch.sigmoid(env_light)
        env_light_img = env_light.expand_as(reflectance)

        soft_mask = 0.9 * mask + 0.1 # 仍使用你的 soft mask，已 resize
        flash_map = flash_map * soft_mask

        illumination = env_light_img + flash_map

        out_img = reflectance * illumination

        # out_img = linear_to_srgb(out_img)

        # result = torch.cat([s,reflectance, illumination, out_img], dim=-1)
        # visualize_tensor(result)
        # visualize_tensor(env_light_img)
        # visualize_tensor(flash_map)
        # visualize_tensor(illumination)
        # visualize_tensor(out_img)
        return out_img

def srgb_to_linear(x):
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4
    )

def linear_to_srgb(x):
    return torch.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * torch.pow(x, 1/2.4) - 0.055
    )


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

