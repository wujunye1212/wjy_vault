import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

"""
    论文地址：https://arxiv.org/abs/2210.04020
    论文题目：Fast-ParC: Capturing Position Aware Global Feature for ConvNets and ViTs（Arxiv 2023）
    中文题目：Fast-ParC：为ConvNets和ViTs捕获全局位置感知特征（Arxiv 2023）
    讲解视频：https://www.bilibili.com/video/BV1G8CFYvEAp/
        位置感知循环卷积（Positional Aware Circular Convolution ,  ParC ）
            理论支撑：通过全局内核卷积捕获长距离信息、位置嵌入以保持位置敏感性。

"""
class ParC_operator(nn.Module):
    def __init__(self, dim, type, global_kernel_size):
        super().__init__()
        self.type = type  # 操作类型，'H'表示水平操作，'W'表示垂直操作
        self.dim = dim  # 输入的通道数

        self.global_kernel_size = global_kernel_size  # 全局卷积核大小
        # 根据操作类型设置卷积核大小
        self.kernel_size = (global_kernel_size, 1) if self.type == 'H' else (1, global_kernel_size)
        # 分组卷积，groups=dim表示每个通道独立卷积
        self.gcc_conv = nn.Conv2d(dim, dim, kernel_size=self.kernel_size, groups=dim)

        # 定义位置编码 (Position Encoding)
        if self.type == 'H':  # 如果是水平操作
            self.pe = nn.Parameter(torch.randn(1, dim, self.global_kernel_size, 1))  # 水平位置编码
        elif self.type == 'W':  # 如果是垂直操作
            self.pe = nn.Parameter(torch.randn(1, dim, 1, self.global_kernel_size))  # 垂直位置编码
        # 初始化位置编码参数，使用截断正态分布
        trunc_normal_(self.pe, std=.02)

    def forward(self, x):
        # 将位置编码添加到输入张量上
        x = x + self.pe.expand(1, self.dim, self.global_kernel_size, self.global_kernel_size)

        if self.type == 'H':  # 如果是水平操作
            # 拼接操作：将输入张量的最后一个高度维度重复一次
            x_cat = torch.cat((x, x[:, :, :-1, :]), dim=2)
        if self.type == 'W':  # 如果是垂直操作
            # 拼接操作：将输入张量的最后一个宽度维度重复一次
            x_cat = torch.cat((x, x[:, :, :, :-1]), dim=3)

        # 对拼接后的张量进行分组卷积
        x = self.gcc_conv(x_cat)
        return x

class ParC_example(nn.Module):
    def __init__(self, dim, global_kernel_size=14):
        super().__init__()
        # 定义水平和垂直的ParC操作，分别作用于输入的前一半和后一半通道
        self.gcc_H = ParC_operator(dim // 2, 'H', global_kernel_size)  # 水平操作
        self.gcc_W = ParC_operator(dim // 2, 'W', global_kernel_size)  # 垂直操作

    def forward(self, x):
        # 将输入张量沿通道维度分成两部分
        x_H, x_W = torch.chunk(x, 2, dim=1)
        # 分别通过水平和垂直的ParC操作
        x_H, x_W = self.gcc_H(x_H), self.gcc_W(x_W)
        # 将处理后的两部分张量在通道维度上拼接
        x = torch.cat((x_H, x_W), dim=1)
        return x

if __name__ == '__main__':
    # 定义一个ParC_example模块，输入通道数为64，全局卷积核大小为56
    block = ParC_example(dim=64, global_kernel_size=56)
    input = torch.rand(3, 64, 56, 56)
    output = block(input)

    print("Input1 shape:", input.shape)
    print("Output shape:", output.shape)

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
