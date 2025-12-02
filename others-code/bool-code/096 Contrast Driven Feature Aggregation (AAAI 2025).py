import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://arxiv.org/pdf/2412.08345
    论文题目：ConDSeg: A General Medical Image Segmentation Framework via Contrast-Driven Feature Enhancement (AAAI 2025)
    中文题目：A2RNet：具有对抗攻击鲁棒性的红外和可见光图像融合网络 (AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV13HrLYuE2K/
        对比驱动特征聚合（Contrast-Driven Feature Aggregation, CDFA）
             理论研究：克服医学图像共现现象导致模型学习错误模式的问题，利用SID解耦的前景和背景特征指导多层次特征融合。
"""
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        # 卷积和批归一化
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)  # 激活函数

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)  # 如果启用激活，则应用ReLU
        return x

# 定义卷积动态特征注意力类
class ContrastDrivenFeatureAggregation(nn.Module):
    def __init__(self, in_c, out_c=128, num_heads=4, kernel_size=3, padding=1, stride=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        dim = out_c
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5  # 缩放因子

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

    def forward(self, x, fg, bg):
        x = self.input_cbr(x)  # 输入特征经过CBR层

        # 调整张量维度顺序
        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        v = self.v(x).permute(0, 3, 1, 2)

        # 前景权重 与 特征x相乘
        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,self.kernel_size * self.kernel_size,-1).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')  # 计算前景注意力
        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)  # 应用前景注意力

        # 背景权重 与 特征x相乘
        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim,self.kernel_size * self.kernel_size,-1).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')  # 计算背景注意力
        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)  # 应用背景注意力

        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)

        out = self.output_cbr(x_weighted_bg)  # 输出特征经过CBR层
        return out

    # 计算注意力
    def compute_attention(self, feature_map, B, H, W, C, feature_type):
        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)  # 应用softmax
        attn = self.attn_drop(attn)
        return attn

    # 应用注意力
    def apply_attention(self, attn, v, B, H, W, C):
        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding,stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted

if __name__ == '__main__':

    cdfa = ContrastDrivenFeatureAggregation(in_c=64, out_c=64)

    x = torch.randn(1, 64, 32, 32)
    fg = torch.randn(1, 64, 32, 32)  # 前景特征图
    bg = torch.randn(1, 64, 32, 32)  # 背景特征图
    output = cdfa(x, fg, bg)

    print("input shape:", x.shape)
    print("fg shape:", fg.shape)
    print("bg shape:", bg.shape)
    print("Output shape:", output.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
