import torch
import numpy as np
import torch.nn as nn
from einops import rearrange

"""
    论文地址：https://arxiv.org/abs/2412.10116
    论文题目：HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection（AAAI 2025）
    中文题目：HS-FPN：用于小目标检测的高频与空间感知特征金字塔网络 （AAAI 2025）
    讲解视频：https://www.bilibili.com/video/BV1zD8kzVE4W/
        空间依赖感知模块（Spatial Dependency Perception Module，SDFM）：
            实际意义：①空间融合缺陷：FPN递归上采样导致上下层特征图中小目标位置错位，仅通过像素级加法融合特征，未建模像素间的空间依赖关系。
                     ②小目标空间信息缺失：小目标特征易被背景噪声掩盖，传统方法无法聚焦局部区域，上层高语义特征与下层细节特征缺乏有效关联，特征表达不完整。
            实现方式：①输入上层特征图A与下层特征图B。
                    ②特征映射生成：通过1×1卷积分别从特征图A生成查询（Q），从特征图B生成键（K）和值（V）。
                    ③特征块划分：将 Q、K、V划分为多个特征块，对每个特征块计算 Q与K的像素级相似度矩阵。
                    ④加权融合：用相似度矩阵对 V 进行加权聚合，生成空间依赖信息特征。⑥输出整合：将特征块按空间位置拼接，与原始输入相加，得到增强后的特征图。
"""

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """自动计算填充大小以保持输出形状与输入相同"""
    # 如果 dilation（膨胀率）大于1，需要调整实际的卷积核大小
    if d > 1:
        # 如果k是整数，计算实际卷积核大小；如果是列表，逐个计算
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # 实际卷积核大小
    # 如果未指定padding（填充），自动计算
    if p is None:
        # 如果k是整数，padding设为k//2；如果是列表，逐个计算
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动填充
    return p

class Conv(nn.Module):
    """标准卷积层，包含参数(ch_in输入通道, ch_out输出通道, kernel卷积核大小, stride步长, padding填充, groups分组数, dilation膨胀率, activation激活函数)"""
    default_act = nn.SiLU()  # 默认激活函数为SiLU

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """用给定参数初始化卷积层，包括激活函数"""
        super().__init__()  # 调用父类nn.Module的初始化方法
        # 创建卷积层，使用autopad函数计算填充
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # 创建批归一化层
        self.bn = nn.BatchNorm2d(c2)
        # 设置激活函数：如果act为True使用默认SiLU；如果是nn.Module实例则使用该实例；否则不使用激活函数
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """对输入张量应用卷积、批归一化和激活函数"""
        # 卷积 -> 批归一化 -> 激活函数的顺序处理
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """执行2D数据的转置卷积（融合模式，不使用批归一化）"""
        return self.act(self.conv(x))

class Spatial_Dependency_Perception_Module(nn.Module):
    """空间依赖感知模块：用于捕获特征图之间的空间依赖关系"""
    def __init__(self,
                 dim=256,  # 输入特征图的通道数
                 patch=None,  # patch大小，用于分割特征图
                 inter_dim=None  # 中间层通道数
                 ):
        super(Spatial_Dependency_Perception_Module, self).__init__()
        self.dim = dim  # 保存输入通道数
        self.inter_dim = inter_dim  # 保存中间层通道数
        # 如果未指定中间层通道数，则默认与输入通道数相同
        if self.inter_dim == None:
            self.inter_dim = dim
        # 定义查询卷积：1x1卷积降低通道数 + 分组归一化
        self.conv_q = nn.Sequential(*[nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False),
                                      nn.GroupNorm(32, self.inter_dim)])
        # 定义键卷积：1x1卷积降低通道数 + 分组归一化
        self.conv_k = nn.Sequential(*[nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False),
                                      nn.GroupNorm(32, self.inter_dim)])
        # 定义softmax层，用于计算注意力权重
        self.softmax = nn.Softmax(dim=-1)
        # 保存patch大小为元组形式
        self.patch_size = (patch, patch)
        # 定义1x1卷积，用于通道数调整
        self.conv1x1 = Conv(self.dim, self.inter_dim, 1)

    def forward(self, x_low, x_high):
        # 获取低分辨率特征图的尺寸信息：b_批次大小, _, h_高度, w_宽度
        b_, _, h_, w_ = x_low.size()

        # 处理查询特征：
        # 1. 通过conv_q卷积
        # 2. 重塑为(batch*h*w, channels, patch1*patch2)的形状
        q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)',
                      p1=self.patch_size[0], p2=self.patch_size[1])
        # 转置使通道维度在后：(batch*h*w, patch1*patch2, channels)
        q = q.transpose(1, 2)  # 示例形状：1,4096,128

        # 处理键特征，与查询特征处理类似
        k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)',
                      p1=self.patch_size[0], p2=self.patch_size[1])

        # 计算注意力权重：查询与键的矩阵乘法
        attn = torch.matmul(q, k)
        # 注意力缩放：除以通道数的平方根，防止梯度消失
        attn = attn / np.power(self.inter_dim, 0.5)
        # 应用softmax获取归一化的注意力权重
        attn = self.softmax(attn)

        # 处理值特征（这里直接使用键特征转置作为值特征）
        v = k.transpose(1, 2)  # 示例形状：1, 1024, 128

        # 应用注意力权重：注意力矩阵乘以值特征
        output = torch.matmul(attn, v)  # 示例形状：1, 4096, 128

        # 将输出重塑回原始特征图形状
        output = rearrange(output.transpose(1, 2).contiguous(),
                           '(b h w) c (p1 p2) -> b c (h p1) (w p2)',
                           p1=self.patch_size[0], p2=self.patch_size[1],
                           h=h_ // self.patch_size[0], w=w_ // self.patch_size[1])

        # 如果输入通道数与中间层通道数不同，调整低分辨率特征图的通道数
        if self.dim != self.inter_dim:
            x_low = self.conv1x1(x_low)

        # 残差连接：注意力输出与低分辨率特征图相加
        return output + x_low

if __name__ == '__main__':
    # 创建随机输入张量1：1个批次，64通道，128x128大小
    input1 = torch.randn(1, 64, 128, 128)
    # 创建随机输入张量2：1个批次，64通道，128x128大小
    input2 = torch.randn(1, 64, 128, 128)
    # 实例化空间依赖感知模块：输入通道64，patch大小8，中间通道32
    sdp = Spatial_Dependency_Perception_Module(64, 8, 64)
    output = sdp(input1, input2)
    print(f"输入张量1形状: {input1.shape}")
    print(f"输入张量2形状: {input2.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")