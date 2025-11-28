import torch
from torch import nn as nn

"""
    论文地址：https://arxiv.org/pdf/2503.01113
    论文题目：SCSegamba: Lightweight Structure-Aware Vision Mamba for CrackSegmentation in Structures (CVPR 2025)
    中文题目：SCSegamba：用于结构裂缝分割的轻量级结构感知视觉曼巴模型 (CVPR 2025)
    讲解视频：https://www.bilibili.com/video/BV1YuZqYyE17/
    门控瓶颈卷积（Gated Bottleneck Convolution, GBC）：
        实际意义：①降低计算成本：传统基于面临着计算资源与分割质量难以平衡的挑战。
                ②增强裂缝特征捕捉能力：不同材料的裂缝在形态和外观上差异显著，现有方法难以有效建模裂缝的形态和纹理。
        实现方式：①通过瓶颈卷积（BottConv），利用低秩近似技术，从高维空间映射到低维空间，降低计算复杂度。
                ②门控机制通过动态调整权重，增强模型捕捉细节的能力，精细化主分支的细粒度特征，从而生成更准确的分割图。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】   
"""

# 用于实现瓶颈卷积结构
class BottConv(nn.Module):
    # 类的初始化函数，接收输入通道数、输出通道数、中间通道数、卷积核大小等参数
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        # 调用父类nn.Module的初始化方法
        super(BottConv, self).__init__()

        # 确保中间通道数至少为2
        mid_channels = max(mid_channels, 2)

        # 第一个逐点卷积层，将输入通道数转换为中间通道数
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        # 深度可分离卷积层，对中间通道数进行卷积操作
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels,bias=False)
        # 第二个逐点卷积层，将中间通道数转换为输出通道数
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    # 定义前向传播方法，描述数据在网络中的流动过程
    def forward(self, x):
        # 数据通过第一个逐点卷积层
        x = self.pointwise_1(x)
        # 经过深度可分离卷积层
        x = self.depthwise(x)
        # 再经过第二个逐点卷积层
        x = self.pointwise_2(x)
        return x

# 实现门控瓶颈卷积结构
class Gated_Bottleneck_Convolution(nn.Module):
    # 类的初始化函数，接收输入通道数和归一化类型作为参数
    def __init__(self, in_channels, norm_type='GN') -> None:
        # 调用父类nn.Module的初始化方法
        super().__init__()

        # 第一个瓶颈卷积模块
        self.proj = BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1)
        # 初始的归一化层，默认为实例归一化
        self.norm = nn.InstanceNorm3d(in_channels)
        # 如果归一化类型为组归一化
        if norm_type == 'GN':
            # 计算组的数量，确保至少为1
            num_groups = max(in_channels // 16, 1)
            # 将归一化层替换为组归一化
            self.norm = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups)
        # 激活函数，使用ReLU
        self.nonliner = nn.ReLU()

        # 第二个瓶颈卷积模块
        self.proj2 = BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1)
        # 第二个归一化层，默认为实例归一化
        self.norm2 = nn.InstanceNorm3d(in_channels)
        # 如果归一化类型为组归一化
        if norm_type == 'GN':
            # 计算组的数量，确保至少为1
            num_groups2 = max(in_channels // 16, 1)
            # 将归一化层替换为组归一化
            self.norm2 = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups2)
        # 激活函数，使用ReLU
        self.nonliner2 = nn.ReLU()

        # 第三个瓶颈卷积模块
        self.proj3 = BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0)
        # 第三个归一化层，默认为实例归一化
        self.norm3 = nn.InstanceNorm3d(in_channels)
        # 如果归一化类型为组归一化
        if norm_type == 'GN':
            # 计算组的数量，确保至少为1
            num_groups3 = max(in_channels // 16, 1)
            # 将归一化层替换为组归一化
            self.norm3 = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups3)
        # 激活函数，使用ReLU
        self.nonliner3 = nn.ReLU()

        # 第四个瓶颈卷积模块
        self.proj4 = BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0)
        # 第四个归一化层，默认为实例归一化
        self.norm4 = nn.InstanceNorm3d(in_channels)
        # 如果归一化类型为组归一化
        if norm_type == 'GN':
            # 计算组的数量，确保至少为1
            num_groups4 = max(in_channels // 16, 1)
            # 将归一化层替换为组归一化
            self.norm4 = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups4)
        # 激活函数，使用ReLU
        self.nonliner4 = nn.ReLU()

    # 定义前向传播方法，描述数据在网络中的流动过程
    def forward(self, x):
        # 保存输入数据作为残差连接
        x_residual = x

        # 左侧 ① 和 ②
        # 数据通过第一个瓶颈卷积模块
        x1_1 = self.proj(x)
        # 经过第一个归一化层
        x1_1 = self.norm(x1_1)
        # 经过第一个激活函数
        x1_1 = self.nonliner(x1_1)

        # 数据通过第二个瓶颈卷积模块
        x1 = self.proj2(x1_1)
        # 经过第二个归一化层
        x1 = self.norm2(x1)
        # 经过第二个激活函数
        x1 = self.nonliner2(x1)

        # 右下①
        # 数据通过第三个瓶颈卷积模块
        x2 = self.proj3(x)
        # 经过第三个归一化层
        x2 = self.norm3(x2)
        # 经过第三个激活函数
        x2 = self.nonliner3(x2)

        # 将x1和x2逐元素相乘
        x = x1 * x2

        # 右上②
        # 数据通过第四个瓶颈卷积模块
        x = self.proj4(x)
        # 经过第四个归一化层
        x = self.norm4(x)
        # 经过第四个激活函数
        x = self.nonliner4(x)

        # 将输出与残差连接相加
        return x + x_residual

if __name__ == '__main__':
    block = Gated_Bottleneck_Convolution(in_channels=32)
    input_tensor = torch.rand(8, 32, 50, 50)
    output_tensor = block(input_tensor)
    print(f'Input size: {input_tensor.size()}')
    print(f'Output size: {output_tensor.size()}')
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")