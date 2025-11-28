from __future__ import annotations
from typing import Optional, Dict
import torch
from torch import Tensor
import torch.nn as nn
'''
2023 CCF BigData 中国计算机协会

本文的核心内容是提出了一个名为 OCA（OrthoNet Convolution Attention，正交通道注意力）模块，
旨在通过正交通道的设计和正交化约束，来提升卷积神经网络的特征提取能力和性能表现。
文中还进一步提出了两个正交通道注意力变体模块 OrthoNet block 和 OrthoNet-MOD block，
这两个模块能够有效减少通道之间的冗余性并增加其独立性，从而提升模型的准确性和计算效率。

理解一下正交通道注意力的模块图
我们的方法包括两个阶段：
阶段 0 是初始化大小与特征层大小匹配的随机滤波器。
然后，我们使用 Gram-Schmidt 过程使这些滤波器进行正交处理。
阶段1 利用这些过滤器来提取挤压向量，并使用 SENet 提出的激励来获得注意力向量。
通过将注意力向量与输入特征相乘，我们计算加权输出特征并添加残差。

主要核心内容包括：
正交通道作用：OCA模块基于正交约束，正交化特征可以减少不同特征通道之间的相关性，防止信息冗余并提升模型的表现。
引入注意力机制的作用：在正交通道的基础上，进一步通过注意力机制来增强特征选择的有效性，提升重要特征的权重。
多尺度特征融合：模块能够在不同尺度上处理输入数据，从而有效增强模型对复杂数据、尤其是图像数据的多维度理解能力。
通过实验验证：本文中通过大量的实验验证，证明OCA模块及其变体在多个计算机视觉任务中的优越性，
             尤其是在提高模型的效率和准确性方面。
通过正交化特征和注意力机制的结合，改进卷积神经网络在多尺度、多维度特征提取上的表现，提升计算机视觉任务中的性能与计算效率。

这个注意力模块适用于所有CV任务，提高模型的效率和性能。
'''

def gram_schmidt(input):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u
    output = []
    for x in input:
        for y in output:
            x = x - projection(y, x)
        x = x/x.norm(p=2)
        output.append(x)
    return torch.stack(output)

def initialize_orthogonal_filters(c, h, w):

    if h*w < c:
        n = c//(h*w)
        gram = []
        for i in range(n):
            gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))
        return torch.cat(gram, dim=0)
    else:
        return gram_schmidt(torch.rand([c, 1, h, w]))
class GramSchmidtTransform(torch.nn.Module):
    instance: Dict[int, Optional[GramSchmidtTransform]] = {}
    constant_filter: Tensor

    @staticmethod
    def build(c: int, h: int):
        if c not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[(c, h)]

    def __init__(self, c: int, h: int):
        super().__init__()
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h, h).view(c, h, h)
        self.register_buffer("constant_filter", rand_ortho_filters.detach())

    def forward(self, x):
        _, _, h, w = x.shape
        _, H, W = self.constant_filter.shape
        if h != H or w != W: x = torch.nn.functional.adaptive_avg_pool2d(x, (H, W))
        return (self.constant_filter * x).sum(dim=(-1, -2), keepdim=True)
class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, FWT: GramSchmidtTransform, input: Tensor):
        #happens once in case of BigFilter
        while input[0].size(-1) > 1:
            input = FWT(input)
        b = input.size(0)
        return input.view(b, -1)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class OCA1(nn.Module):
    def __init__(self, inplanes,planes, height, stride=1, downsample=None):
        super(OCA1, self).__init__()
        self._process: nn.Module = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.conv1x1 = nn.Conv2d(planes*4,inplanes,1)
        self.downsample = downsample
        self.stride = stride

        self.planes = planes
        self._excitation = nn.Sequential(
            nn.Linear(in_features=4 * planes, out_features=round(planes / 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(planes / 4), out_features=4 * planes, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention =Attention()
        self.F_C_A = GramSchmidtTransform.build(4 * planes, height)
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self._process(x)
        compressed = self.OrthoAttention(self.F_C_A, out)
        b, c = out.size(0), out.size(1)
        attention = self._excitation(compressed).view(b, c, 1, 1)
        attention = attention * out
        attention = self.conv1x1(attention)
        attention += residual
        activated = torch.relu(attention)
        return activated

class OCA2(nn.Module):
    def __init__(self, inplanes, planes, height, stride=1, downsample=None):
        super(OCA2, self).__init__()

        self._preprocess: nn.Module = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        self._scale: nn.Module = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.conv1x1 = nn.Conv2d(planes * 4, inplanes, 1)
        self.downsample = downsample
        self.stride = stride
        self.planes = planes

        self._excitation = nn.Sequential(
            nn.Linear(in_features=planes, out_features=round(planes / 16), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(planes / 16), out_features=planes, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention()
        self.F_C_A = GramSchmidtTransform.build(planes, height)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        preprocess_out = self._preprocess(x)
        compressed = self.OrthoAttention(self.F_C_A, preprocess_out)
        b, c = preprocess_out.size(0), preprocess_out.size(1)
        attention = self._excitation(compressed).view(b, c, 1, 1)
        attentended = attention * preprocess_out
        scale_out = self._scale(attentended)
        scale_out = self.conv1x1(scale_out)
        scale_out += residual
        activated = torch.relu(scale_out)
        return activated

# 输入 B C H W   输出 B C H W
if __name__ == '__main__':
    # 创建输入张量
    input = torch.randn(1, 64, 128, 128)
    # 定义 BasicBlock 模块
    # block = OCA1(inplanes=64,planes=64, height=128)
    block = OCA2(inplanes=64,planes=64, height=128)
    # 前向传播
    output = block(input)
    # 打印输入和输出的形状
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")