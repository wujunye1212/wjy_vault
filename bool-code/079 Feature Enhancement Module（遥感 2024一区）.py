import torch
import torch.nn as nn

"""
    论文地址：https://ieeexplore.ieee.org/abstract/document/10423050
    论文题目：FFCA-YOLO for Small Object Detection in Remote Sensing Images（2024 一区TOP）
    中文题目：遥感图像小目标检测的FFCA-YOLO （2024 一区TOP）
    讲解视频：https://www.bilibili.com/video/BV18FrZYWEZ8/
        特征增强模块（Feature Enhancement Module，FEM）：
            理论支撑：通过多分支卷积结构和空洞卷积操作，更有效地捕捉小目标周围的上下文信息，增强了网络对于小目标的特征提取能力。
"""

class BasicConv(nn.Module):
    # 基础卷积块
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes  # 输出通道数
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, # 卷积层
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None # 批归一化层
        self.relu = nn.ReLU(inplace=True) if relu else None # ReLU激活层

    def forward(self, x):
        x = self.conv(x) # 卷积操作
        if self.bn is not None:
            x = self.bn(x) # 批归一化
        if self.relu is not None:
            x = self.relu(x) # ReLU激活
        return x

class FEM(nn.Module):
    # 特征增强模块（Feature Enhancement Module）
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale # 缩放因子
        self.out_channels = out_planes # 输出通道数
        inter_planes = in_planes // map_reduce # 中间通道数

        self.branch0 = nn.Sequential( # 分支0
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride), # 1x1卷积
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False) # 3x3卷积，无ReLU
        )

        self.branch1 = nn.Sequential( # 分支1
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), # 1x1卷积
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)), # 1x3卷积
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)), # 3x1卷积
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False) # 3x3空洞卷积，dilation=5，无ReLU
        )

        self.branch2 = nn.Sequential( # 分支2
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), # 1x1卷积
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)), # 3x1卷积
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)), # 1x3卷积
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False) # 3x3空洞卷积，dilation=5，无ReLU
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False) # 1x1卷积，无ReLU
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False) #  shortcut连接，1x1卷积，无ReLU
        self.relu = nn.ReLU(inplace=False) # ReLU激活

    def forward(self, x):
        x0 = self.branch0(x) # 分支0输出
        x1 = self.branch1(x) # 分支1输出
        x2 = self.branch2(x) # 分支2输出
        out = torch.cat((x0, x1, x2), 1) # 特征拼接
        out = self.ConvLinear(out) # 1x1卷积

        short = self.shortcut(x) # shortcut输出

        out = out * self.scale + short # 缩放并与shortcut相加

        out = self.relu(out) # ReLU激活
        return out

if __name__ == '__main__':
    batch_size = 4 # 批量大小
    in_channels = 64 # 输入通道数
    height = 32 # 高度
    width = 32 # 宽度

    x = torch.randn(batch_size, in_channels, height, width) # 创建随机输入张量
    swa = FEM(in_channels,in_channels) # 创建FEM模块实例

    print("Input shape:", x.shape) # 打印输入形状
    out_swa = swa(x) # 前向传播
    print("Output shape:", out_swa.shape) # 打印输出形状

