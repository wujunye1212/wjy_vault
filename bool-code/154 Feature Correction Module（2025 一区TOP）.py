import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ # 来自timm库的截断正态分布初始化函数
import math

"""
    论文地址：https://ieeexplore.ieee.org/abstract/document/10786275
    论文题目：CFFormer: A Cross-Fusion Transformer Framework for the Semantic Segmentation of Multisource Remote Sensing Images （TGRS 2025）
    中文题目：CFFormer：一种用于多源遥感图像语义分割的交叉融合Transformer框架（TGRS 2025）
    讲解视频：https://www.bilibili.com/video/BV1u2JuzrEfH/
    特征修正模块（Feature Correction Module，FCM）：
        实际意义：①多模态数据差异：不同传感器的成像原理和物理特性差异，导致多模态数据存在显著特征不一致性（如光谱差异、空间分辨率差异）。
                ②局部空间特征与全局语义特征的不匹配：不同模态在同一区域的表现方式不一致。
                ③互补信息利用不充分：传统方法直接融合多模态特征时，易因模态差异导致互补信息被噪声掩盖。
        实现方式：先通过空间权重对齐局部结构，再用通道权重增强全局语义互补，减少模态差异并突出有效信息。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

# 通道权重计算模块（用于生成通道维度的注意力权重）
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        """
        :param dim: 输入特征的通道数维度（通常为C）
        :param reduction: 通道压缩比例（用于减少MLP层的参数量）
        """
        super(ChannelWeights, self).__init__()  # 调用父类初始化方法
        self.dim = dim  # 保存通道维度参数
        # 自适应平均池化：将空间维度压缩为1x1（保留通道维度信息）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 自适应最大池化：同样压缩空间维度，捕捉通道维度的最大值信息
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # MLP多层感知机：用于将统计特征映射为通道权重
        self.mlp = nn.Sequential(
            # 输入维度：6*dim（因为要拼接avg/std/max三种统计量，各2*dim）
            # 中间层维度：6*dim // reduction（通过reduction压缩参数量）
            nn.Linear(self.dim * 6, self.dim * 6 // reduction),
            nn.ReLU(inplace=True),  # 激活函数增加非线性
            # 输出维度：2*dim（对应两个输入特征的通道权重）
            nn.Linear(self.dim * 6 // reduction, self.dim * 2),
            nn.Sigmoid()  # Sigmoid将输出归一化到[0,1]区间作为权重
        )

    def forward(self, x1, x2):
        """
        :param x1: 输入特征1，形状(B, C, H, W)
        :param x2: 输入特征2，形状(B, C, H, W)
        :return: 通道权重，形状(2, B, C, 1, 1)
        """
        B, _, H, W = x1.shape  # 获取批次大小B和空间尺寸H/W
        x = torch.cat((x1, x2), dim=1)  # 拼接两个特征（通道维度拼接，形状变为(B, 2C, H, W)）

        # 计算三种统计量（捕捉通道维度的全局信息）
        avg = self.avg_pool(x).view(B, self.dim * 2)  # 平均池化后展平：(B, 2C)
        std = torch.std(x, dim=(2, 3), keepdim=True).view(B, self.dim * 2)  # 空间维度标准差：(B, 2C)
        max = self.max_pool(x).view(B, self.dim * 2)  # 最大池化后展平：(B, 2C)

        y = torch.cat((avg, std, max), dim=1)  # 拼接三种统计量：(B, 6C)

        y = self.mlp(y).view(B, self.dim * 2, 1)  # MLP处理后恢复维度：(B, 2C, 1)

        # 调整维度顺序以匹配后续融合操作：
        # 原始形状(B, 2C, 1) -> 重塑为(2, B, C, 1, 1)（对应两个特征的通道权重）
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights


# 空间权重计算模块（用于生成空间维度的注意力权重）
class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        """
        :param dim: 输入特征的通道数维度（通常为C）
        :param reduction: 通道压缩比例
        """
        super(SpatialWeights, self).__init__()
        self.dim = dim  # 保存通道维度参数
        # MLP使用卷积实现（保持空间维度信息）
        self.mlp = nn.Sequential(
            # 输入通道：2C（拼接后的特征）-> 输出通道：C//reduction（压缩通道）
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),  # 激活函数
            # 输出通道：2（对应两个特征的空间权重）
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid()  # 归一化到[0,1]作为空间权重
        )

    def forward(self, x1, x2):
        """
        :param x1: 输入特征1，形状(B, C, H, W)
        :param x2: 输入特征2，形状(B, C, H, W)
        :return: 空间权重，形状(2, B, 1, H, W)
        """
        B, _, H, W = x1.shape  # 获取批次大小和空间尺寸
        x = torch.cat((x1, x2), dim=1)  # 通道维度拼接特征：(B, 2C, H, W)

        # 通过MLP生成空间权重（保持空间维度HxW）
        spatial_weights = self.mlp(x)  # 形状：(B, 2, H, W)

        # 调整维度顺序以匹配后续融合操作：
        # 原始形状(B, 2, H, W) -> 重塑为(2, B, 1, H, W)（对应两个特征的空间权重）
        spatial_weights = spatial_weights.reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_weights


# 特征融合模块（主模块，整合空间和通道注意力）
class FCM(nn.Module):
    def __init__(self, dim, reduction=1, eps=1e-8):
        """
        :param dim: 输入特征的通道数维度（通常为C）
        :param reduction: 通道压缩比例（传递给子模块）
        :param eps: 防止除零的极小值
        """
        super(FCM, self).__init__()
        # 可训练的融合权重参数（初始化为[1,1]，训练中自动学习两个特征的重要性）
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps  # 保存极小值参数

        # 初始化空间权重和通道权重子模块
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)

        # 应用权重初始化函数（对所有子模块的参数进行初始化）
        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        """ 自定义参数初始化策略（针对不同层类型） """
        if isinstance(m, nn.Linear):
            # 线性层权重：截断正态分布（timm库的实现）
            trunc_normal_(m.weight, std=.02)
            # 线性层偏置：若存在则初始化为0
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # 层归一化：权重初始化为1，偏置初始化为0
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # 卷积层权重：根据输出通道数计算正态分布标准差（He初始化）
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups  # 分组卷积时调整
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 卷积层偏置：若存在则初始化为0
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        # 对可训练权重应用ReLU确保非负性，并归一化（保证权重和为1）
        weights = nn.ReLU()(self.weights)  # 形状(2,)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)  # 归一化：(2,)
        # 计算空间注意力权重（形状(2, B, 1, H, W)）
        spatial_weights = self.spatial_weights(x1, x2)

        # 第一步融合：基于空间权重的特征交互
        # x1_1 = x1 + (fuse_weights[0] * 空间权重[1] * x2)：用x2的空间权重增强x1
        x1_1 = x1 + fuse_weights[0] * spatial_weights[1] * x2
        # x2_1 = x2 + (fuse_weights[0] * 空间权重[0] * x1)：用x1的空间权重增强x2
        x2_1 = x2 + fuse_weights[0] * spatial_weights[0] * x1

        # 计算通道注意力权重（形状(2, B, C, 1, 1)）
        channel_weights = self.channel_weights(x1_1, x2_1)

        # 第二步融合：基于通道权重的特征交互
        main_out = x1_1 + fuse_weights[1] * channel_weights[1] * x2_1
        aux_out = x2_1 + fuse_weights[1] * channel_weights[0] * x1_1

        return main_out, aux_out

if __name__ == "__main__":
    x1 = torch.randn(1, 32, 50, 50)
    x2 = torch.randn(1, 32, 50, 50)
    fcm = FCM(dim=32)
    main_out, aux_out = fcm(x1, x2)
    print(f"输入张量1形状: {x1.shape}")
    print(f"输入张量2形状: {x2.shape}")
    print(f"输出张量1形状: {main_out.shape}")
    print(f"输出张量2形状: {aux_out.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")