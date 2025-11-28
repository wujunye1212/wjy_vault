import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C3

'''
来自ACCV 2024 论文 含二次创新
即插即用注意力模块： LocalAttention 局部增强注意力模块  （简称LIA 基于重要的局部注意力模块）
含二次创新模块 CLAttention 通道局部增强注意力模块  效果优于LIA ，可以冲SCI二区或是三区，B会或C会

LIA 模块（Local Importance-based Attention）在论文中旨在实现一种高效的2阶注意力机制，
用以改善超分辨率任务中的特征选择和信息交互。它的主要作用包括：
1.高效的进行注意力计算：LIA 模块通过计算输入特征的局部重要性（Local Importance），
并结合通道门（Channel Gate）来重新校准注意力图。这种设计实现了2阶信息交互，
但避免了传统2阶注意力（如自注意力）的高计算复杂度。
2.低延迟和性能平衡：LIA 模块通过在下采样的特征图上计算局部重要性，
并利用简单的操作（例如卷积和双线性插值），在提升模型表达能力的同时，显著降低了计算延迟。
3.增强的特征权重化：LIA 使用局部重要性和通道门的组合，
使得模块能够自适应地增强有用的特征，同时弱化无用的特征，从而提高了特征选择的精度。

LIA模块通过高效的设计实现了2阶注意力特性的同时，大幅降低了计算开销，使得其特别适合应用于低延迟、高效率的超分辨率任务中。

这个模块适用于：图像超分辨率任务、目标检测、图像分割、图像分类、
             轻量级图像处理、图像增强（如去噪、去模糊）、图像修复等计算机视觉CV任务通用的即插即用模块
'''

class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool
class LocalAttention(nn.Module):
    ''' attention based on local importance'''

    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        g = self.gate(x[:, :1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return x * w * g  # (w + g) #self.gate(x, w)

class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

'''
二次创新模块 CLAttention 通道局部增强注意力模块  可以冲SCI二区或是三区，B会或C会

CLAttention 通道局部增强注意力模块 内容介绍：

首先执行通道注意力机制。它对每个通道进行全局平均池化，
然后通过1D卷积来捕捉通道之间的交互信息。这种方法避免了降维问题，
确保模型能够有效地聚焦在最相关的通道特征上。
然后利用LIA局部注意力模块的作用如下：
1.高效的注意力计算：LIA 模块通过计算输入特征的局部重要性（Local Importance），
并结合通道门（Channel Gate）来重新校准注意力图。这种设计实现了2阶信息交互，
但避免了传统2阶注意力（如自注意力）的高计算复杂度。
2.低延迟和性能平衡：LIA 模块通过在下采样的特征图上计算局部重要性，
并利用简单的操作（例如卷积和双线性插值），在提升模型表达能力的同时，显著降低了计算延迟。
3.增强的特征权重化：LIA 使用局部重要性和通道门的组合，
使得模块能够自适应地增强有用的特征，同时弱化无用的特征，从而提高了特征选择的精度。

'''
class CLAttention(nn.Module):
    def __init__(self, channels, f=16):
        super().__init__()
        self.ca= channel_att(channels)
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        ''' forward '''
        x = self.ca(x) #在这里添加一个通道注意力模块，有效地捕捉通道之间的交互信息，促进重要局部特征表示。
        # interpolate the heat map
        g = self.gate(x[:, :1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return x * w * g  # (w + g) #self.gate(x, w)

if __name__ == "__main__":
    input = torch.randn(1, 32, 64, 64)
    LA = LocalAttention(32)
    output = LA(input)
    print('input_size:', input.size())
    print('output_size:', output.size())

    CLA = CLAttention(32)
    output = CLA(input)
    print('二次创新_CLA_input_size:', input.size())
    print('二次创新_CLA_output_size:', output.size())