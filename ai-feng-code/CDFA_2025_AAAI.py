import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C3

'''
来自AAAI 2025年的顶会   来自医学图像分割任务的2025年AAAI顶会论文
即插即用模块：CDFA 对比度驱动特征聚合模块 （特征融合模块）  这个模块名字听起来就比较新颖

五步法搞定学会书写顶会顶刊论文摘要，能够大大提高发小论文的中稿率!!!!

医学图像分割在临床决策制定、治疗计划和疾病跟踪中起着重要作用。 ---交代了：本文做的任务，点明这篇论文主题

然而，它仍然面临两大挑战。一方面，医学图像中前景和背景之间常常存在“软边界”，
且光照不足和对比度低进一步降低了图像中前景和背景的区分度。另一方面，医学图像中普遍存在共现现象
，学习这些特征会对模型的判断产生误导。    ----交代了：我们发现某种问题，我们提出一种方法可以很好的解决这个问题

为解决这些挑战，我们提出了一种通用框架，称为对比驱动的医学图像分割。 ---交代了：本文的创新方法

首先，我们开发了一种称为一致性增强的对比训练策略，旨在提高编码器在各种光照和对比度场景下的鲁棒性，
使模型即使在恶劣环境中也能提取高质量特征。其次，1.我们引入了语义信息解耦模块，能够将编码器中的特征解耦为前景、背景和不确定性区域，
并在训练过程中逐渐获取减少不确定性的能力。2.对比驱动的特征聚合模块随后对比前景和背景特征，以指导多级特征融合和关键特征增强，进一步区分待分割的实体。
3.我们还提出了一种尺寸感知解码器，以解决解码器的尺度单一性问题，它能够准确定位图像中不同大小的实体，从而避免对共现特征的错误学习。
                                                            ---简单介绍本文的具体创新点 3个  
                                                            
在三个场景下的五个医学图像数据集上进行的大量实验证明了我们方法的性能，证明了其先进性和对各种医学图像分割场景的普遍适用性。
                                                            ---通过广泛实验表明，我们的创新方法是有效的，具有普适性。

CDFA模块的作用:
1.多级别特征融合与增强：CDFA模块通过融合来自编码器的多级特征图，并结合前景和背景特征图，实现特征的多级融合与增强。
         增强后的特征表示有助于模型更好地区分待分割的实体和背景，提高分割的准确性和鲁棒性。
2.增强关键特征表示：通过对比前景和背景特征，CDFA模块能够凸显关键特征，抑制非关键特征。
         这使得模型在分割过程中更加关注于重要的前景区域，减少背景干扰，提高分割精度。
3.解决共现现象：在医学图像中，共现现象（不同实体同时出现）是常见的挑战。CDFA模块通过增强关键特征，
          使得模型能够更好地应对共现现象，避免学习到错误的共现特征，从而提高了分割的准确性。
          
这个模块适用于：医学图像分割，目标检测，语义分割，图像分类，图像增强等所有计算机视觉CV任务通用的模块
'''

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x
"""Decouple Layer"""
class DecoupleLayer(nn.Module):
    def __init__(self, in_c=1024, out_c=256):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        # f_uc = self.cbr_uc(x)
        # return f_fg, f_bg, f_uc
        return f_fg, f_bg

class CDFA(nn.Module):
    def __init__(self, in_c, out_c=128, num_heads=4, kernel_size=3, padding=1, stride=1,attn_drop=0., proj_drop=0.):
        super().__init__()
        dim = out_c
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.dcp = DecoupleLayer(in_c,dim)
    def forward(self, x,fg, bg):

        x = self.input_cbr(x)

        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)

        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')

        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim,
                                                                               self.kernel_size * self.kernel_size,
                                                                               -1).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')

        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)

        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)

        out = self.output_cbr(x_weighted_bg)

        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):

        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):

        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # 实例化 CDFA模块
    cdfa = CDFA(in_c=64,out_c=64)
    x = torch.randn(1,64,32,32)
    fg = torch.randn(1, 64, 32, 32) #前景特征图
    bg = torch.randn(1, 64, 32, 32) #背景特征图
    # 这个模块前向传播输入张量 x, fg, 和 bg。
    output = cdfa(x,fg,bg)
    # 打印输出张量的形状
    print("input shape:", x.shape)
    print("Output shape:", output.shape)
