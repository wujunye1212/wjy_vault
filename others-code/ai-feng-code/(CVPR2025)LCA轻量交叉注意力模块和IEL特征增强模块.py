import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
'''
来自CVPR 2025顶会 低光图像增强任务
即插即用模块: LCA 轻量交叉注意力模块
IEL 特征增强模块   CDL 颜色去噪模块

主要内容：
低光图像增强（Low-Light Image Enhancement，LLIE）是计算机视觉中的一项关键任务，从光照不足的图像中恢复详细的视觉信息。
许多现有的 LLIE 方法基于标准的 sRGB 颜色空间，但由于 sRGB 本身对颜色高度敏感，这些方法往往会产生色偏差和亮度伪影。
尽管将图像转换到 HSV（色调、饱和度、亮度）颜色空间可以在一定程度上解决亮度问题，但这种转换也会引入显著的红色和黑色噪声伪影。

为了解决这些问题，我们提出了一种新的低光图像增强颜色空间——HVI（Horizontal/Vertical-Intensity），
该空间由极化 HS（色调-饱和度）映射和可学习的强度映射函数构成。前者通过缩小红色坐标的距离来消除红色伪影，
而后者则通过压缩低光区域来去除黑色伪影。此外，我们进一步提出了一种新的颜色与强度解耦网络（CIDNet），
该网络能够在 HVI 颜色空间中学习不同光照条件下的精准光度映射函数，从而充分利用色彩信息和强度信息。

在多个基准数据集和消融实验的综合评估中，结果表明，我们提出的 HVI 颜色空间结合 CIDNet，
在 10 个数据集上均超越了现有的最先进方法。
----------------------------------------------

LCA（Lighten Cross-Attention）模块的作用:
是增强亮度信息和颜色信息之间的特征交互，以提高低光图像的增强效果。
它采用交叉注意力机制，让亮度分支和颜色分支相互提供信息，从而避免单独处理导致的亮度偏差或颜色失真。
通过计算亮度特征对颜色特征的注意力分布，以及颜色特征对亮度特征的关注程度，该模块能够更精准地调整光照强度，
并优化颜色一致性，使增强后的图像更加自然。

IEL/CDL（Intensity Enhance Layer）模块的作用:
是提升图像的亮度表现，同时减少光照不均和低光区域的噪声。它基于 Retinex 理论，将图像的亮度信息和反射信息进行分离，
并通过非线性变换增强亮度特征，使低光区域的细节得以保留，同时避免过度增强导致的伪影或光晕。
通过自适应调整亮度，该模块可以有效改善低光场景下的图像质量，使图像整体亮度更均衡、细节更丰富。

适用于：低光图像增强，图像去噪，图像恢复，图像去雨雪/雾/模糊，目标检测，图像分割，等所有CV任务通用的即插即用模块
'''
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
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

# Cross Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# Intensity Enhancement Layer
class IEL(nn.Module): # 强度增强层
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


class CDL(nn.Module):  #颜色去噪层
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(CDL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x

# Lightweight Cross Attention
class LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(LCA, self).__init__()
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x)) + x #建议对增强后的特征使用残差连接操作，恢复丢失的细节特征
        return x

# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    module =  LCA(dim=64,num_heads=4)
    input_x = torch.randn(1, 64, 32, 32)
    input_y = torch.randn(1, 64, 32, 32)
    output_tensor = module(input_x,input_y)
    print('LCA_Input size:', input_x.size())  # 打印输入张量的形状
    print('LCA_Output size:', output_tensor.size())  # 打印输出张量的形状

    IEL = IEL(dim=64) # IEL=CDL
    output_tensor = IEL(input_x)
    print('IEL_Input size:', input_x.size())  # 打印输入张量的形状
    print('IEL_Output size:', output_tensor.size())  # 打印输出张量的形状

