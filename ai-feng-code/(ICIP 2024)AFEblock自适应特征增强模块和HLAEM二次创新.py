import torch
from torch import nn
from timm.models.layers import DropPath
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
'''

AFEBlock模块的作用：
AFEBlock模块是一种自适应特征增强块，旨在处理复杂场景中的语义分割任务，特别是在杂乱背景和半透明对象存在的情况下。
它通过并行地结合空间上下文模块（SCM）和特征细化模块（FRM）来增强特征，从而改善语义分割的性能。

AFEBlock模块的原理：
空间上下文模块（SCM）：SCM利用较大的卷积核来增加感受野，从而捕获更广泛的空间上下文信息，以应对场景中的尺度变化。
特征细化模块（FRM）：FRM受到图像锐化和对比度增强的启发，旨在捕获低频上下文并强调图像中的区域，同时突出高频细节。
它通过特定的网络结构（如深度可分离卷积和上采样）对输入特征进行细化处理。
并行处理与融合：SCM和FRM的输出通过1x1卷积层和ConvMLP（卷积多层感知机）进行融合，以进一步增强特征表示。
这种并行处理方式使得AFEBlock能够同时利用空间上下文信息和特征细化结果，从而提高语义分割的准确性。

'''

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class FeatureRefinementModule(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, down_kernel=5, down_stride=4):
        super().__init__()

        self.lconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.hconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.norm1 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.down = nn.Conv2d(in_dim, in_dim, kernel_size=down_kernel, stride=down_stride, padding=down_kernel // 2,
                              groups=in_dim)
        self.proj = nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, stride=1, padding=0)

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape

        dx = self.down(x)
        udx = F.interpolate(dx, size=(H, W), mode='bilinear', align_corners=False)
        lx = self.norm1(self.lconv(self.act(x * udx)))
        hx = self.norm2(self.hconv(self.act(x - udx)))

        out = self.act(self.proj(torch.cat([lx, hx], dim=1)))

        return out
class AFE(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim // 2, 1, padding=0)
        self.proj2 = nn.Conv2d(dim, dim, 1, padding=0)

        self.ctx_conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=7, padding=3, groups=4)

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")

        self.enhance = FeatureRefinementModule(in_dim=dim // 2, out_dim=dim // 2, down_kernel=3, down_stride=2)

        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.norm1(self.act(self.dwconv(x)))
        x = self.norm2(self.act(self.proj1(x)))
        ctx = self.norm3(self.act(self.ctx_conv(x)))  #SCM模块

        enh_x = self.enhance(x)                       #FRM模块
        x = self.act(self.proj2(torch.cat([ctx, enh_x], dim=1)))
        return x
class AFEBlock(nn.Module):
    def __init__(self, dim, drop_path=0.1, expan_ratio=4,kernel_size=3, use_dilated_mlp=True):
        super().__init__()

        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        if use_dilated_mlp:
            self.mlp = AtrousMLP(dim=dim, mlp_ratio=expan_ratio)
        else:
            self.mlp = MLP(dim=dim, mlp_ratio=expan_ratio)
        self.attn = AFE(dim, kernel_size=kernel_size)

        self.drop_path_1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        inp_copy = x
        x = self.layer_norm1(inp_copy)
        x = self.drop_path_1(self.attn(x))
        out = x + inp_copy

        x = self.layer_norm2(out)
        x = self.drop_path_2(self.mlp(x))
        out = out + x

        return out
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, use_dcn=False):
        super().__init__()

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x
class AtrousMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos1 = nn.Conv2d(dim * mlp_ratio, dim * 2, 3, padding=1, groups=dim * 2)
        self.pos2 = nn.Conv2d(dim * mlp_ratio, dim * 2, 3, padding=2, dilation=2, groups=dim * 2)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.act(self.fc1(x))
        x1 = self.act(self.pos1(x))
        x2 = self.act(self.pos2(x))
        x_a = torch.cat([x1, x2], dim=1)
        x = self.fc2(x_a)

        return x

# 二次创新
class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),
            norm(in_dim),
            act()
        )

    def forward(self, x):
        mid = self.in_conv(x)

        out = mid
        # print(out.shape)

        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)

        out = self.out_conv(out)

        return out
class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out,kernel_size=3, stride=1, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out,kernel_size,stride)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out,kernel_size=3, stride=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# 冲sci 一区或二区
'''
HLAEM二次创新模块

HLAEM高低频特征自适应模块的介绍：

1.高频信息： 通常反映图像中的局部细节（例如边缘、纹理等），在各自视觉任务中非常重要。
因此，通过大卷积核的深度卷积、反向瓶颈设计以及残差连接，在轻量化的同时高效提取这些高频特征，
并通过密集连接机制增强高频信息的细节。

2.低频信息： 通常代表图像的整体结构，主要用于恢复图像的大尺度信息，比如图像的轮廓和背景。
在这方面低频信息起着关键作用。解决图像细节边缘特征缺失的问题，提高模型捕捉物体边界的能力。
使用3×3平均池化和1×1卷积从输入图像中提取多尺度的边缘信息。通过边缘增强器（EE），
在每个尺度上强化边缘感知，突出物体的关键边界。提取的多尺度边缘信息与主分支的特征融合，
最终提升低频特征的精细度。
'''
class HLAEM(nn.Module): #高低频特征自适应模块
    def __init__(self, dim):
        super(HLAEM, self).__init__()
        n_feats = dim
        self.down = nn.AvgPool2d(kernel_size=2)
        self.cmunextblock = CMUNeXtBlock(n_feats,n_feats)
        self.meem = MEEM(dim,dim)
        self.alise1 = nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0)  # one_module(n_feats)
        self.alise2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)  # one_module(n_feats)
        self.att = CALayer(n_feats)
    def forward(self, x):
        low = self.down(x)
        high = x - F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)
        lowf = self.meem(low)
        highfeat =  self.cmunextblock(high)
        lowfeat = F.interpolate(lowf, size=x.size()[-2:], mode='bilinear', align_corners=True)
        out = self.alise2(self.att(self.alise1(torch.cat([highfeat, lowfeat], dim=1)))) + x
        return out
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.randn(1, 32, 64, 64) #随机生成一张输入图片张量
    # 初始化AFEBlock模块并设定通道维度
    AFEBlock  = AFEBlock(dim=32)
    output = AFEBlock(input)  # 进行前向传播
    # 输出结果的形状
    print("AFEBlock_输入张量的形状：", input.shape)
    print("AFEBlock_输出张量的形状：", output.shape)

    # 二次创新模块，初始化HLAEM模块并设定通道维度
    HLAEM  = HLAEM(dim=32)
    output = HLAEM(input)  # 进行前向传播
    # 输出结果的形状
    print("HLAEM_输入张量的形状：", input.shape)
    print("HLAEM_输出张量的形状：", output.shape)
