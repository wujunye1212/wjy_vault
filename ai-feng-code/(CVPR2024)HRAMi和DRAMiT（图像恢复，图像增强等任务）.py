import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import torch
import math
from timm.models.fx_features import register_notrace_function
import warnings
warnings.filterwarnings('ignore')
'''
     用于轻量级图像恢复的相互注意力混合Transformer   CVPR 2024顶会
     
尽管最近的许多工作在图像修复 （IR） 领域取得了进展，但它们经常受到参数过多的影响。
另一个问题是，大多数基于 Transformer 的 IR 方法只关注局部或全局特征，导致感受野有限或参数不足问题。
为了解决这些问题，我们提出了一种轻量级网络，即（RAMiT）。

RAMiT网络的核心由两个关键模块组成：HRAMi 和 DRAMiT 

HRAMi 和 DRAMiT 两个模块的原理及作用：
HRAMi模块：
原理：HRAMi模块通过对层次结构中不同阶段特征图进行混合，帮助网络恢复在下采样过程中丢失的像素级信息。
它会将来自不同层次的多尺度注意力进行上采样，随后将这些注意力进行融合，从而补偿在层次结构中的信息丢失，
并利用语义层级的信息来增强表示能力。
作用：该模块增强了图像的恢复效果，尤其是细致区域的边界信息，使得网络能够处理多尺度的图像信息，
并在低级和高级特征间建立联系，提高图像复原的准确性。

DRAMiT模块 ：
原理：DRAMiT模块包含空间自注意力(SPSA)和通道自注意力(CHSA)两种注意力机制，并且这两种机制并行计算。
不同的注意力头被分配给空间和通道自注意力，输出的注意力结果相互帮助，克服彼此的缺点。
通过这种双维度的自注意力混合机制，DRAMiT模块能够同时捕捉到局部和全局的图像依赖性。

作用：DRAMiT模块解决了仅依赖局部或全局特征时感受野不足或参数不足的问题，
确保模型能够在轻量级的前提下捕捉图像中的边缘和纹理等细节信息，从而提升图像恢复任务的效果。

适用于：图像恢复、图像增强、超分辨率图像、低光增强、图像去噪、图像去雾、图像去雨滴等所有CV任务通用即插即用模块
'''
class ShallowModule(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, stride=1):
        super(ShallowModule, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride, kernel_size // 2)

    def forward(self, x):
        return self.conv(x)

    def flops(self, resolutions):
        return resolutions[0] * resolutions[1] * self.kernel_size * self.kernel_size * self.in_chans * self.out_chans
class QKVProjection(nn.Module):
    def __init__(self, dim, num_head, qkv_bias=True):
        super(QKVProjection, self).__init__()
        self.dim = dim
        self.num_head = num_head

        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=qkv_bias)

    def forward(self, x):
        B, C, H, W = x.size()
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b (l c) h w -> b l c h w', l=self.num_head)
        return qkv

    def flops(self, resolutions):
        return resolutions[0] * resolutions[1] * 1 * 1 * self.dim * 3 * self.dim
def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None,
                                                   :]  # 2, Wh*Ww, Wh*Ww (xaxis matrix & yaxis matrix)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
class SpatialSelfAttention(nn.Module):
    def __init__(self, dim, num_head, total_head, window_size=8, shift=0, attn_drop=0.0, proj_drop=0.0, helper=True):
        super(SpatialSelfAttention, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.total_head = total_head
        self.window_size = window_size
        self.window_area = window_size ** 2
        self.shift = shift
        self.helper = helper

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_head))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size, window_size))

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Conv2d(dim * num_head, dim * num_head, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, qkv, ch=None):
        B, L, C, H, W = qkv.size()
        # window shift
        if self.shift > 0:
            qkv = torch.roll(qkv, shifts=(-self.shift, -self.shift), dims=(-2, -1))

        # window partition
        q, k, v = rearrange(qkv, 'b l c (h wh) (w ww) -> (b h w) l (wh ww) c',
                            wh=self.window_size, ww=self.window_size).chunk(3, dim=-1)  # [B_, L1, hw, C/L] respectively
        if ch is not None and self.helper:  # [B, C, H, W]
            if self.shift > 0:
                ch = torch.roll(ch, shifts=(-self.shift, -self.shift), dims=(-2, -1))
            ch = rearrange(ch, 'b (l c) (h wh) (w ww) -> (b h w) l (wh ww) c',
                           l=self.total_head - self.num_head, wh=self.window_size,
                           ww=self.window_size)  # [B_, L1, hw, C/L]
            ch = torch.mean(ch, dim=1, keepdim=True)  # head squeeze [B_, 1, hw, C/L]
            v = v * ch  # [B_, L1, hw, C/L]

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(2, -1)  # [B_, L1, hw, hw]
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        attn = attn + self._get_rel_pos_bias()
        attn = self.attn_drop(F.softmax(attn, dim=-1))

        x = attn @ v  # [B_, L1, hw, C/L]

        # window unpartition + head merge
        x = window_unpartition(x, (H, W), self.window_size)  # [B, L1*C/L, H, W]
        x = self.proj_drop(self.proj(x))

        # window reverse shift
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(-2, -1))

        return x

    def flops(self, resolutions):
        H, W = resolutions
        num_wins = H // self.window_size * W // self.window_size
        flops = self.num_head * H * W * self.dim if self.helper else 0  # v = v*ch
        flops += num_wins * self.num_head * self.window_area * self.dim * self.window_area  # attn = Q@K^T
        flops += num_wins * self.num_head * self.window_area * self.window_area * self.dim  # attn@V
        flops += H * W * 1 * 1 * self.num_head * self.dim * self.num_head * self.dim  # self.proj
        return flops
@register_notrace_function
def window_unpartition(x, resolutions, window_size):
    return rearrange(x, '(b h w) l (wh ww) c -> b (l c) (h wh) (w ww)',
                     h=resolutions[0] // window_size, w=resolutions[1] // window_size, wh=window_size)
class ChannelSelfAttention(nn.Module):
    def __init__(self, dim, num_head, total_head, attn_drop=0.0, proj_drop=0.0, helper=True):
        super(ChannelSelfAttention, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.total_head = total_head
        self.helper = helper

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Conv2d(dim * num_head, dim * num_head, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, qkv, sp=None):
        B, L, C, H, W = qkv.size()

        q, k, v = rearrange(qkv, 'b l c h w -> b l c (h w)').chunk(3, dim=-2)  # [B, L2, C/L, HW]
        if sp is not None and self.helper:
            sp = torch.mean(sp, dim=1, keepdim=True)  # channel squeeze # [B, 1, H, W]
            sp = rearrange(sp, 'b (l c) h w -> b l c (h w)', l=1)  # [B, 1, 1, HW]
            v = v * sp  # [B, L2, C/L, HW]

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(2, -1)  # [B, L2, C/L, C/L]
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # [B, L2, C/L, HW]

        # head merge
        x = rearrange(x, 'b l c (h w) -> b (l c) h w', h=H)  # [B, L2*C/L, H, W]
        x = self.proj_drop(self.proj(x))  # [B, L2*C/L, H, W]

        return x

    def flops(self, resolutions):
        H, W = resolutions
        flops = self.num_head * self.dim * H * W if self.helper else 0  # v = v*sp
        flops += self.num_head * self.dim * H * W * self.dim  # attn = Q@K^T
        flops += self.num_head * self.dim * self.dim * H * W  # attn@V
        flops += H * W * 1 * 1 * self.num_head * self.dim * self.num_head * self.dim  # self.proj
        return flops
class ReshapeLayerNorm(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(ReshapeLayerNorm, self).__init__()

        self.dim = dim
        self.norm = norm_layer(dim)

    def forward(self, x):
        B, C, H, W = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H)
        return x

    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        flops += H * W * self.dim
        return flops
class NoLayer(nn.Identity):
    def __init__(self):
        super(NoLayer, self).__init__()
    def flops(self, resolutions):
        return 0
    def forward(self, x, **kwargs):
        return x.flatten(1,2)
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
class DRAMiT(nn.Module):  # Reciprocal Attention Transformer Block
    def __init__(self, dim, num_head=8, chsa_head_ratio=0.5, window_size=8, shift=0, head_dim=None, qkv_bias=True, mv_ver=1,
                 hidden_ratio=2.0, act_layer=nn.GELU, norm_layer=ReshapeLayerNorm, attn_drop=0.0, proj_drop=0.0,
                 drop_path=0.0, helper=True,
                 mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(DRAMiT, self).__init__()

        self.dim = dim
        self.num_head = num_head
        self.window_size = window_size
        self.chsa_head = int(num_head * chsa_head_ratio)
        self.shift = shift
        self.helper = helper

        self.qkv_proj = QKVProjection(dim, num_head, qkv_bias=qkv_bias)
        self.sp_attn = SpatialSelfAttention(dim // num_head, num_head - self.chsa_head, num_head,
                                            window_size, shift, attn_drop, proj_drop,
                                            helper) if num_head - self.chsa_head != 0 else NoLayer()
        self.ch_attn = ChannelSelfAttention(dim // num_head, self.chsa_head, num_head, attn_drop, proj_drop,
                                            helper) if self.chsa_head != 0 else NoLayer()
        if mv_ver == 1:
            self.mobivari = MobiVari1(dim, 3, 1, act=mv_act)
        elif mv_ver == 2:
            self.mobivari = MobiVari2(dim, 3, 1, act=mv_act, out_dim=None, exp_factor=exp_factor,
                                      expand_groups=expand_groups)

        self.norm1 = norm_layer(dim)

        self.ffn = FeedForward(dim, hidden_ratio, act_layer=act_layer)
        self.norm2 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, sp_=None, ch_=None):
        B, C, H, W = x.size()

        # QKV projection + head split
        qkv = self.qkv_proj(x)  # [B, L, C, H, W]

        # SP-SA / CH-SA
        sp = self.sp_attn(qkv[:, :self.num_head - self.chsa_head], ch=ch_)  # [B, L1*C/L, H, W]
        ch = self.ch_attn(qkv[:, self.num_head - self.chsa_head:], sp=sp_)  # [B, L2*C/L, H, W]
        attn0 = self.mobivari(torch.cat([sp, ch], dim=1))  # merge [B, C, H, W]
        attn = self.drop_path(self.norm1(attn0)) + x  # LN, skip connection [B, C, H, W]

        # FFN
        out = self.drop_path(self.norm2(self.ffn(attn))) + attn  # FFN, LN, skip connection [B, C, H, W]
        return out
        # return out, sp, ch, attn0

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_ratio, act_layer=nn.GELU, bias=True, drop=0.0):
        super(FeedForward, self).__init__()

        self.dim = dim
        self.hidden_ratio = hidden_ratio

        self.hidden = nn.Conv2d(dim, int(dim * hidden_ratio), 1, bias=bias)
        self.drop1 = nn.Dropout(drop)
        self.out = nn.Conv2d(int(dim * hidden_ratio), dim, 1, bias=bias)
        self.drop2 = nn.Dropout(drop)
        self.act = act_layer()

    def forward(self, x):
        return self.drop2(self.out(self.drop1(self.act(self.hidden(x)))))

    def flops(self, resolutions):
        H, W = resolutions
        flops = 2 * H * W * 1 * 1 * self.dim * self.dim * self.hidden_ratio  # self.hidden + self.out
        return flops
class MobiVari1(nn.Module):  # MobileNet v1 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None):
        super(MobiVari1, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.out_dim = out_dim or dim

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, stride, kernel_size // 2, groups=dim)
        self.pw_conv = nn.Conv2d(dim, self.out_dim, 1, 1, 0)
        self.act = act()

    def forward(self, x):
        out = self.act(self.pw_conv(self.act(self.dw_conv(x)) + x))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * self.kernel_size * self.kernel_size * self.dim + H * W * 1 * 1 * self.dim * self.out_dim  # self.dw_conv + self.pw_conv
        return flops
class MobiVari2(MobiVari1):  # MobileNet v2 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None, exp_factor=1.2, expand_groups=4):
        super(MobiVari2, self).__init__(dim, kernel_size, stride, act, out_dim)
        self.expand_groups = expand_groups
        expand_dim = int(dim * exp_factor)
        expand_dim = expand_dim + (expand_groups - expand_dim % expand_groups)
        self.expand_dim = expand_dim

        self.exp_conv = nn.Conv2d(dim, self.expand_dim, 1, 1, 0, groups=expand_groups)
        self.dw_conv = nn.Conv2d(expand_dim, expand_dim, kernel_size, stride, kernel_size // 2, groups=expand_dim)
        self.pw_conv = nn.Conv2d(expand_dim, self.out_dim, 1, 1, 0)

    def forward(self, x):
        x1 = self.act(self.exp_conv(x))
        out = self.pw_conv(self.act(self.dw_conv(x1) + x1))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * 1 * 1 * (self.dim // self.expand_groups) * self.expand_dim  # self.exp_conv
        flops += H * W * self.kernel_size * self.kernel_size * self.expand_dim  # self.dw_conv
        flops += H * W * 1 * 1 * self.expand_dim * self.out_dim  # self.pw_conv
        return flops
class HRAMi(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, mv_ver=1, mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(HRAMi, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        if mv_ver == 1:
            self.mobivari = MobiVari1(dim + dim // 4 + dim // 16 + dim, kernel_size, stride, act=mv_act, out_dim=dim)
        elif mv_ver == 2:
            self.mobivari = MobiVari2(dim + dim // 4 + dim // 16 + dim, kernel_size, stride, act=mv_act, out_dim=dim,exp_factor=2., expand_groups=1)

    def forward(self, attn_list):
        for i, attn in enumerate(attn_list[:-1]):
            attn = F.pixel_shuffle(attn, 2 ** i)
            x = attn if i == 0 else torch.cat([x, attn], dim=1)

        x = torch.cat([x, attn_list[-1]], dim=1)
        x = self.mobivari(x)
        return x

    def flops(self, resolutions):
        return self.mobivari.flops(resolutions)


if __name__ == '__main__':
    # 创建DRAMiT模块的实例
    DRAMiT = DRAMiT(dim=64)
    # 创建输入张量：B C H W
    input = torch.randn(1, 64, 32, 32)
    # 执行DRAMiT模块的前向传播
    output = DRAMiT(input)
    # 输出输入和输出张量的形状
    print("DRAMiT_Input size:" ,input.size())
    print("DRAMiT_Output size:" ,output.size())


    HRAMi = HRAMi(dim=64)
    input = [
                torch.randn(1, 64, 64, 64),  # stage1  C H W
                torch.randn(1, 64, 32, 32),  # stage2  C H/2 W/2
                torch.randn(1, 64, 16, 16),  # stage3  C H/4 W/4
                torch.randn(1, 64, 64, 64)   # stage4  C H W
            ]
    # 执行HRAMi模块的前向传播
    output = HRAMi(input)   #output的尺度和stage4特征图尺度相同  C H W
    print('HRAMi_input size:', input[3].size())
    print('HRAMi_output size:', output.size())

