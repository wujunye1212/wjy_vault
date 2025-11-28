import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_ as trunc_normal_init


class CrissCrossContrastAttention(nn.Module):
    """
    十字交叉对比注意力 (Criss-Cross Contrast Attention, CCCA) 模块

    本模块是为极小目标检测设计的全新轻量化注意力机制。
    1. 局部对比度分支: 通过计算原始特征与不同尺度模糊版本之间的差异，高效地
       定位图像中对比度强烈的区域（潜在缺陷）。
    2. 十字交叉上下文分支: 使用高效的条纹卷积来捕捉每个像素点在其所在行和列的
       长距离依赖关系。
    3. 引导式融合: 利用第一阶段生成的“对比度图”来加权增强第二阶段的上下文特征，
       实现对潜在目标区域的精准放大。

    参数:
        in_channels (int): 输入特征图的通道数。
        inter_channels_ratio (float): 在十字交叉分支中，中间通道的缩放比例。
        num_heads (int): 在十字交叉分支中，注意力头的数量。
    """

    def __init__(self, in_channels, inter_channels_ratio=0.25, num_heads=8):
        super(CrissCrossContrastAttention, self).__init__()

        self.in_channels = in_channels
        inter_channels = int(in_channels * inter_channels_ratio)

        # --- 局部对比度分支 ---
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.contrast_merge = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # --- 十字交叉上下文分支 ---
        self.num_heads = num_heads
        self.inter_channels = inter_channels
        assert inter_channels % num_heads == 0, "inter_channels must be divisible by num_heads"

        self.norm = nn.BatchNorm2d(in_channels)
        # 可学习的垂直和水平条纹卷积核
        self.kernel_v = nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1))
        self.kernel_h = nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7))
        trunc_normal_init(self.kernel_v, std=0.001)
        trunc_normal_init(self.kernel_h, std=0.001)

        # --- 融合与控制 ---
        self.gamma = nn.Parameter(torch.zeros(1))

    def _act_dn(self, x):
        """ 模拟注意力的激活函数 """
        x_shape = x.shape
        h, w = x_shape[2], x_shape[3]
        x = x.reshape(x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1)
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)
        x = x.reshape(x_shape[0], self.inter_channels, h, w)
        return x

    def forward(self, x):
        """
        前向传播函数
        """
        identity = x

        # --- 1. 局部对比度分支 ---
        contrast3 = x - self.pool3(x)
        contrast5 = x - self.pool5(x)
        combined_contrast = torch.cat([contrast3, contrast5], dim=1)
        contrast_map = self.contrast_merge(torch.abs(combined_contrast))  # 形状: (B, 1, H, W)

        # --- 2. 十字交叉上下文分支 ---
        x_norm = self.norm(x)

        # 垂直上下文
        x_v = F.conv2d(x_norm, self.kernel_v, bias=None, stride=1, padding=(3, 0))
        x_v = self._act_dn(x_v)
        x_v = F.conv2d(x_v, self.kernel_v.transpose(0, 1), bias=None, stride=1, padding=(3, 0))

        # 水平上下文
        x_h = F.conv2d(x_norm, self.kernel_h, bias=None, stride=1, padding=(0, 3))
        x_h = self._act_dn(x_h)
        x_h = F.conv2d(x_h, self.kernel_h.transpose(0, 1), bias=None, stride=1, padding=(0, 3))

        context_features = x_v + x_h

        # --- 3. 引导式融合 ---
        # 使用对比度图对上下文特征进行加权
        enhanced_features = context_features * contrast_map

        # 使用可学习参数 gamma 进行缩放，并与原始输入进行残差连接
        out = identity + self.gamma * enhanced_features

        return out


if __name__ == '__main__':
    # --- 使用示例 ---
    in_channels = 64
    input_feature_map = torch.randn(4, in_channels, 128, 128)

    # 在主特征图上创建一个极小的缺陷 (一个3x3的亮斑)
    defect = torch.ones(3, 3) * 5.0
    input_feature_map[0, 10, 62:65, 62:65] = defect

    # 初始化CCCA模块
    ccca_block = CrissCrossContrastAttention(in_channels=in_channels)

    # 将特征图输入模块
    output_feature_map = ccca_block(input_feature_map)

    # 打印输入和输出的形状
    print(f"输入特征图形状: {input_feature_map.shape}")
    print(f"输出特征图形状: {output_feature_map.shape}")

    # 验证模块是否增强了缺陷区域的特征
    input_energy_before = torch.sum(input_feature_map[0, :, 62:65, 62:65] ** 2)
    output_energy_after = torch.sum(output_feature_map[0, :, 62:65, 62:65] ** 2)

    print(f"\n缺陷区域输入能量: {input_energy_before.item():.2f}")
    print(f"缺陷区域输出能量: {output_energy_after.item():.2f}")

    if output_energy_after > input_energy_before:
        print("\n成功: CCCA模块增强了缺陷区域的特征信号！")
    else:
        print("\n注意: 模块未增强特征信号，请检查参数或实现。")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ECA(nn.Module):
    """
    高效通道注意力 (Efficient Channel Attention)
    灵感来源: ECA-Net (CVPR 2020), HCF-Net (arxiv 2024)
    """

    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        # 动态计算一维卷积的核大小
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, x_spatial_attn=None):
        """
        如果提供了x_spatial_attn，则执行空间引导的通道注意力。
        """
        if x_spatial_attn is not None:
            # 空间引导：在池化前，用空间注意力图对x进行加权
            x = x * x_spatial_attn

        # Squeeze
        out = self.pool(x)
        # Reshape for 1D convolution
        out = out.view(x.size(0), 1, x.size(1))
        # Excitation
        out = self.conv(out)
        # Reshape for multiplication
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out


class ContrastSpatialAttention(nn.Module):
    """
    对比度空间注意力
    灵感来源: B2CNet (2024 Top)
    """

    def __init__(self, in_channels, reduction_ratio=4):
        super(ContrastSpatialAttention, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.contrast_merge = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算对比度
        contrast = x - self.pool(x)
        # 生成空间注意力图
        spatial_attn_map = self.contrast_merge(torch.abs(contrast))
        return spatial_attn_map


class ReciprocalChannelSpatialAttention(nn.Module):
    """
    往复式通道-空间注意力 (Reciprocal Channel-Spatial Attention, RCSA)

    本模块是为极小目标检测设计的全新轻量化注意力机制。
    1. 通道引导空间: 先用ECA计算初步的通道权重，引导空间注意力聚焦于重要通道。
    2. 空间引导通道: 再用上一步生成的空间注意力图，反过来引导ECA的计算，
       得到更精准的最终通道权重。
    3. 双重增强: 将最终的通道和空间注意力同时应用于原始特征。

    参数:
        in_channels (int): 输入特征图的通道数。
    """

    def __init__(self, in_channels):
        super(ReciprocalChannelSpatialAttention, self).__init__()

        self.in_channels = in_channels

        # --- 初始化专家模块 ---
        self.channel_expert = ECA(in_channels)
        self.spatial_expert = ContrastSpatialAttention(in_channels)

        # --- 控制 ---
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        前向传播函数
        """
        identity = x

        # --- 1. 第一程: 通道引导空间 ---
        # 计算初步的通道注意力
        initial_channel_attn = self.channel_expert(x)
        # 应用通道注意力，得到通道精炼后的特征
        x_channel_refined = x * initial_channel_attn
        # 在通道精炼后的特征上计算空间注意力
        spatial_attn = self.spatial_expert(x_channel_refined)

        # --- 2. 第二程: 空间引导通道 (往复步骤) ---
        # 利用上一步生成的空间注意力图，来引导通道注意力的计算
        final_channel_attn = self.channel_expert(x, x_spatial_attn=spatial_attn)

        # --- 3. 双重增强与融合 ---
        # 将最终的通道和空间注意力同时应用于原始特征
        enhanced_features = identity * final_channel_attn * spatial_attn

        # 使用可学习参数 gamma 进行缩放，并与原始输入进行残差连接
        out = identity + self.gamma * enhanced_features

        return out


if __name__ == '__main__':
    # --- 使用示例 ---
    in_channels = 64
    input_feature_map = torch.randn(4, in_channels, 128, 128)

    # 在主特征图上创建一个极小的缺陷 (一个3x3的亮斑)
    defect = torch.ones(3, 3) * 5.0
    input_feature_map[0, 10, 62:65, 62:65] = defect

    # 初始化RCSA模块
    rcsa_block = ReciprocalChannelSpatialAttention(in_channels=in_channels)

    # 将特征图输入模块
    output_feature_map = rcsa_block(input_feature_map)

    # 打印输入和输出的形状
    print(f"输入特征图形状: {input_feature_map.shape}")
    print(f"输出特征图形状: {output_feature_map.shape}")

    # 验证模块是否增强了缺陷区域的特征
    input_energy_before = torch.sum(input_feature_map[0, :, 62:65, 62:65] ** 2)
    output_energy_after = torch.sum(output_feature_map[0, :, 62:65, 62:65] ** 2)

    print(f"\n缺陷区域输入能量: {input_energy_before.item():.2f}")
    print(f"缺陷区域输出能量: {output_energy_after.item():.2f}")

    if output_energy_after > input_energy_before:
        print("\n成功: RCSA模块增强了缺陷区域的特征信号！")
    else:
        print("\n注意: 模块未增强特征信号，请检查参数或实现。")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ECA(nn.Module):
    """
    高效通道注意力 (Efficient Channel Attention)
    灵感来源: ECA-Net (CVPR 2020), HCF-Net (arxiv 2024)
    """

    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        # 动态计算一维卷积的核大小
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, x_spatial_attn=None):
        """
        如果提供了x_spatial_attn，则执行空间引导的通道注意力。
        """
        if x_spatial_attn is not None:
            # 空间引导：在池化前，用空间注意力图对x进行加权
            x = x * x_spatial_attn

        # Squeeze
        out = self.pool(x)
        # Reshape for 1D convolution
        out = out.view(x.size(0), 1, x.size(1))
        # Excitation
        out = self.conv(out)
        # Reshape for multiplication
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out


class ContrastSpatialAttention(nn.Module):
    """
    对比度空间注意力
    灵感来源: B2CNet (2024 Top)
    """

    def __init__(self, in_channels, reduction_ratio=4):
        super(ContrastSpatialAttention, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.contrast_merge = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算对比度
        contrast = x - self.pool(x)
        # 生成空间注意力图
        spatial_attn_map = self.contrast_merge(torch.abs(contrast))
        return spatial_attn_map


class ReciprocalChannelSpatialAttention(nn.Module):
    """
    往复式通道-空间注意力 (Reciprocal Channel-Spatial Attention, RCSA)

    本模块是为极小目标检测设计的全新轻量化注意力机制。
    1. 通道引导空间: 先用ECA计算初步的通道权重，引导空间注意力聚焦于重要通道。
    2. 空间引导通道: 再用上一步生成的空间注意力图，反过来引导ECA的计算，
       得到更精准的最终通道权重。
    3. 双重增强: 将最终的通道和空间注意力同时应用于原始特征。

    参数:
        in_channels (int): 输入特征图的通道数。
    """

    def __init__(self, in_channels):
        super(ReciprocalChannelSpatialAttention, self).__init__()

        self.in_channels = in_channels

        # --- 初始化专家模块 ---
        self.channel_expert = ECA(in_channels)
        self.spatial_expert = ContrastSpatialAttention(in_channels)

        # --- 控制 ---
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        前向传播函数
        """
        identity = x

        # --- 1. 第一程: 通道引导空间 ---
        # 计算初步的通道注意力
        initial_channel_attn = self.channel_expert(x)
        # 应用通道注意力，得到通道精炼后的特征
        x_channel_refined = x * initial_channel_attn
        # 在通道精炼后的特征上计算空间注意力
        spatial_attn = self.spatial_expert(x_channel_refined)

        # --- 2. 第二程: 空间引导通道 (往复步骤) ---
        # 利用上一步生成的空间注意力图，来引导通道注意力的计算
        final_channel_attn = self.channel_expert(x, x_spatial_attn=spatial_attn)

        # --- 3. 双重增强与融合 ---
        # 将最终的通道和空间注意力同时应用于原始特征
        enhanced_features = identity * final_channel_attn * spatial_attn

        # 使用可学习参数 gamma 进行缩放，并与原始输入进行残差连接
        out = identity + self.gamma * enhanced_features

        return out


if __name__ == '__main__':
    # --- 使用示例 ---
    in_channels = 64
    input_feature_map = torch.randn(4, in_channels, 128, 128)

    # 在主特征图上创建一个极小的缺陷 (一个3x3的亮斑)
    defect = torch.ones(3, 3) * 5.0
    input_feature_map[0, 10, 62:65, 62:65] = defect

    # 初始化RCSA模块
    rcsa_block = ReciprocalChannelSpatialAttention(in_channels=in_channels)

    # 将特征图输入模块
    output_feature_map = rcsa_block(input_feature_map)

    # 打印输入和输出的形状
    print(f"输入特征图形状: {input_feature_map.shape}")
    print(f"输出特征图形状: {output_feature_map.shape}")

    # 验证模块是否增强了缺陷区域的特征
    input_energy_before = torch.sum(input_feature_map[0, :, 62:65, 62:65] ** 2)
    output_energy_after = torch.sum(output_feature_map[0, :, 62:65, 62:65] ** 2)

    print(f"\n缺陷区域输入能量: {input_energy_before.item():.2f}")
    print(f"缺陷区域输出能量: {output_energy_after.item():.2f}")

    if output_energy_after > input_energy_before:
        print("\n成功: RCSA模块增强了缺陷区域的特征信号！")
    else:
        print("\n注意: 模块未增强特征信号，请检查参数或实现。")

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolaritySensitiveContrastAttention(nn.Module):
    """
    极性敏感对比度注意力 (Polarity-Sensitive Contrast Attention, PSCA)

    本模块是为极小目标检测设计的全新轻量化注意力机制。
    1. 双极对比度探测: 通过计算原始特征与模糊版本的差异，并利用ReLU解耦，
       独立地定位正向对比（亮异常）和负向对比（暗异常）的区域。
    2. 特征精炼: 使用一个高效的倒置瓶颈块来增强特征的局部细节表达。
    3. 双极门控融合: 利用两个独立的极性注意力门，分别增强对应类型的特征，
       然后融合，实现对潜在目标区域的精准放大。

    参数:
        in_channels (int): 输入特征图的通道数。
        refine_growth_ratio (float): 在特征精炼分支中，通道的扩张比例。
    """

    def __init__(self, in_channels, refine_growth_ratio=2.0):
        super(PolaritySensitiveContrastAttention, self).__init__()

        self.in_channels = in_channels

        # --- 双极对比度探测分支 ---
        # 用于生成模糊背景的平均池化层
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # 共享的轻量级卷积网络，用于从极性图中生成注意力门
        self.gate_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # --- 局部特征精炼分支 ---
        hidden_dim = int(in_channels * refine_growth_ratio)
        self.refine_block = nn.Sequential(
            # 1x1 卷积升维
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # 3x3 深度卷积进行空间混合
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # 1x1 卷积降维
            nn.Conv2d(hidden_dim, in_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # --- 控制 ---
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        前向传播函数
        """
        identity = x

        # --- 1. 双极对比度探测 ---
        # 计算核心对比度图
        contrast = x - self.pool(x)

        # 解耦为正、负两个极性图
        pos_contrast = F.relu(contrast)
        neg_contrast = F.relu(-contrast)

        # 分别生成正、负两个注意力门
        gate_pos = self.gate_generator(pos_contrast)  # 形状: (B, 1, H, W)
        gate_neg = self.gate_generator(neg_contrast)  # 形状: (B, 1, H, W)

        # --- 2. 局部特征精炼 ---
        refined_features = self.refine_block(x)

        # --- 3. 双极门控融合 ---
        # 分别用两个门控信号增强精炼后的特征
        enhanced_pos = refined_features * gate_pos
        enhanced_neg = refined_features * gate_neg

        # 将两种增强后的特征相加，得到最终的增强信号
        enhanced_features = enhanced_pos + enhanced_neg

        # 使用可学习参数 gamma 进行缩放，并与原始输入进行残差连接
        out = identity + self.gamma * enhanced_features

        return out


if __name__ == '__main__':
    # --- 使用示例 ---
    in_channels = 64
    input_feature_map = torch.randn(4, in_channels, 128, 128)

    # 在主特征图上创建两种极小缺陷
    # 1. 亮斑缺陷 (正向对比)
    bright_defect = torch.ones(3, 3) * 5.0
    input_feature_map[0, 10, 30:33, 30:33] = bright_defect

    # 2. 暗坑缺陷 (负向对比)
    dark_defect = torch.ones(3, 3) * -5.0
    input_feature_map[1, 20, 90:93, 90:93] = dark_defect

    # 初始化PSCA模块
    psca_block = PolaritySensitiveContrastAttention(in_channels=in_channels)

    # 将特征图输入模块
    output_feature_map = psca_block(input_feature_map)

    # 打印输入和输出的形状
    print(f"输入特征图形状: {input_feature_map.shape}")
    print(f"输出特征图形状: {output_feature_map.shape}")

    # 验证模块是否增强了两种缺陷区域的特征
    input_energy_bright_before = torch.sum(input_feature_map[0, :, 30:33, 30:33] ** 2)
    output_energy_bright_after = torch.sum(output_feature_map[0, :, 30:33, 30:33] ** 2)

    input_energy_dark_before = torch.sum(input_feature_map[1, :, 90:93, 90:93] ** 2)
    output_energy_dark_after = torch.sum(output_feature_map[1, :, 90:93, 90:93] ** 2)

    print(f"\n亮斑缺陷区域输入能量: {input_energy_bright_before.item():.2f}")
    print(f"亮斑缺陷区域输出能量: {output_energy_bright_after.item():.2f}")

    print(f"\n暗坑缺陷区域输入能量: {input_energy_dark_before.item():.2f}")
    print(f"暗坑缺陷区域输出能量: {output_energy_dark_after.item():.2f}")

    if output_energy_bright_after > input_energy_bright_before and output_energy_dark_after > input_energy_dark_before:
        print("\n成功: PSCA模块同时增强了两种极性缺陷的特征信号！")
    else:
        print("\n注意: 模块未增强特征信号，请检查参数或实现。")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- 核心组件 1: SimAM (无参数注意力) ---
class simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


# --- 专家分支 1: 空间对比度专家 ---
class SpatialContrastBranch(nn.Module):
    def __init__(self, in_channel):
        super(SpatialContrastBranch, self).__init__()
        self.simam1 = simam_module()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.simam2 = simam_module()

    def forward(self, x):
        x_enhanced = self.simam1(x)
        edge = x_enhanced - self.avg_pool(x_enhanced)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        out = weight * x_enhanced + x_enhanced
        return self.simam2(out)


# --- 专家分支 2: 通道注意力专家 (ECA) ---
class ChannelAttentionBranch(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ChannelAttentionBranch, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return x * out.expand_as(x)


# --- 专家分支 3: 频率分析专家 ---
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FrequencyAnalysisBranch(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyAnalysisBranch, self).__init__()
        self.process_magnitude = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=in_channels),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0))
        self.process_phase = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=in_channels),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0))

    def forward(self, x):
        b, c, h, w = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        magnitude, phase = torch.abs(x_freq), torch.angle(x_freq)
        magnitude += self.process_magnitude(magnitude)
        phase += self.process_phase(phase)
        real, imag = magnitude * torch.cos(phase), magnitude * torch.sin(phase)
        refined_freq = torch.complex(real, imag)
        return torch.fft.irfft2(refined_freq, s=(h, w), norm='backward')


# --- 融合中心: 深度融合模块 ---
class DeepFusionModule(nn.Module):
    def __init__(self, inc, outc):
        super(DeepFusionModule, self).__init__()
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(inc, outc, 1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
        self.Conv = nn.Sequential(
            nn.Conv2d(outc, outc, 3, 1, 1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat1, feat2, feat3):
        cat = torch.cat([feat1, feat2, feat3], dim=1)
        cat = self.Conv_1(cat) + feat1 + feat2 + feat3
        c = self.Conv(cat) + cat
        c = self.relu(c) + feat1  # 再次强化最重要的特征（假设feat1是空间对比度）
        return c


# --- 最终的注意力模块 ---
class MultiDimensionalContrastAttentionNetwork(nn.Module):
    """
    多维对比度注意力网络 (Multi-Dimensional Contrast Attention Network, MDCAN)

    本模块是为极小目标检测设计的终极版，融合了多种先进思想：
    1. 空间对比度专家: 使用SimAM和特征差分高效地定位图像中的对比度区域。
    2. 通道注意力专家: 使用高效的ECA模块，对特征通道进行加权。
    3. 频率分析专家: 使用幅相解耦网络智能地净化和增强全局频率特征。
    4. 深度特征融合: 使用先进的DFEM思想，利用密集的残差连接来智能地融合三个专家分支的特征。

    参数:
        in_channels (int): 输入特征图的通道数。
    """

    def __init__(self, in_channels):
        super(MultiDimensionalContrastAttentionNetwork, self).__init__()
        self.in_channels = in_channels

        # --- 并行三专家引擎 ---
        self.spatial_expert = SpatialContrastBranch(in_channels)
        self.channel_expert = ChannelAttentionBranch(in_channels)
        self.frequency_expert = FrequencyAnalysisBranch(in_channels)

        # --- DFEM 融合模块 ---
        self.deep_fusion_center = DeepFusionModule(inc=in_channels * 3, outc=in_channels)

        # --- 控制 ---
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x

        # --- 1. 并行专家处理 ---
        spatial_features = self.spatial_expert(x)
        channel_features = self.channel_expert(x)
        frequency_features = self.frequency_expert(x)

        # --- 2. 深度智能融合 ---
        fused_features = self.deep_fusion_center(spatial_features, channel_features, frequency_features)

        # --- 3. 残差连接 ---
        out = identity + self.gamma * fused_features

        return out


if __name__ == '__main__':
    # --- 使用示例 ---
    in_channels = 64
    input_feature_map = torch.randn(4, in_channels, 128, 128)

    # 在主特征图上创建一个极小的缺陷
    defect = torch.ones(3, 3) * 5.0
    input_feature_map[0, 10, 62:65, 62:65] = defect

    # 初始化MDCAN模块
    mdcan_block = MultiDimensionalContrastAttentionNetwork(in_channels=in_channels)

    # 将特征图输入模块
    output_feature_map = mdcan_block(input_feature_map)

    # 打印输入和输出的形状
    print(f"输入特征图形状: {input_feature_map.shape}")
    print(f"输出特征图形状: {output_feature_map.shape}")

    # 验证模块是否增强了缺陷区域的特征
    input_energy_before = torch.sum(input_feature_map[0, :, 62:65, 62:65] ** 2)
    output_energy_after = torch.sum(output_feature_map[0, :, 62:65, 62:65] ** 2)

    print(f"\n缺陷区域输入能量: {input_energy_before.item():.2f}")
    print(f"缺陷区域输出能量: {output_energy_after.item():.2f}")

    if output_energy_after > input_energy_before:
        print("\n成功: MDCAN模块增强了缺陷区域的特征信号！")
    else:
        print("\n注意: 模块未增强特征信号，请检查参数或实现。")
