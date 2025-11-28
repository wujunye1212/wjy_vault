import torch
import torch.nn as nn
'''
来自CVPR2025 顶会
即插即用卷积模块： GBConv门控块卷积
带来一个二次创新模块: GSConv 门控激活卷积

一、GBConv模块的作用
GBConv模块的主要作用是增强网络对空间与通道特征的建模能力，通过一种动态的门控机制控制信息流动，
从而提高网络的特征表示能力。这对于提升模型在图像识别、分割等视觉任务中的性能尤为关键。
论文中的实验证明，在相同的参数规模下，引入GBConv模块的网络在多个基准数据集
（如ImageNet、ADE20K、COCO）上都显著优于基线模型，说明该模块在提升网络表现方面具有重要作用(大家都可以去直接使用) 。

二、GBConv模块的原理
GBConv模块的基本思想是将标准卷积操作与门控机制相结合。
其具体工作原理如下：
1.平行出来：输入特征图首先通过两个并行分支，分别执行不同的操作，之后再融合：
    一支用于执行标准卷积（提取局部特征）
    另一支用于生成门控权重（控制信息通行）
2.门控机制（Gating）：另一支通过轻量级操作（如1×1卷积+激活函数）生成一个门控图，表示当前通道或空间位置的重要性。
3.特征调制：将门控图与原始卷积特征进行逐元素相乘，实现信息筛选和调制，从而突出重要特征、抑制无关信息。
4.残差连接与融合：GBConv模块通常引入残差连接，将输入和输出进行融合，以保持稳定训练并增强特征表达。
5.可嵌入性强：GBConv模块设计轻量、结构灵活，可以方便地插入至主干网络（如ResNet、MobileNet）中，提升其性能。

GSConv模块适合：图像分割，语义分割，目标检测等所有CV任务通用卷积模块
'''
class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x
def get_norm_layer(norm_type, channels, num_groups):
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm3d(channels)
class GBConv(nn.Module):
    def __init__(self, in_channels, norm_type='GN'):
        super(GBConv, self).__init__()

        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        x = x1 * x2
        x = self.block4(x)

        return x + residual

#二次创模块:GSConv 门控激活卷积模块
class GSConv(nn.Module):
    def __init__(self, in_channels, norm_type='GN'):
        super(GSConv, self).__init__()

        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.psi = nn.Sequential(  #增加一个门控机制，对self.block3提取后的特征，使用sigmod激活函数生成一个权重
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        gate =self.psi(x2)#使用 self.psi
        x = x1 * gate
        x = self.block4(x)

        return x + residual

# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    input = torch.randn(1,32,64, 64)  # 创建一个形状为 (1,32,64, 64)
    GBConv = GBConv(32)
    output = GBConv(input)  # 通过GBConv模块计算输出
    print('Ai缝合怪永久更新中—GBConv_Input size:', input.size())  # 打印输入张量的形状
    print('Ai缝合怪永久更新中—GBConv_Output size:', output.size())  # 打印输出张量的形状

    input = torch.randn(1,32,64, 64)  # 创建一个形状为 (1,32,64, 64)
    GSConv = GSConv(32)
    output = GSConv(input)  # 通过GSConv门控激活卷积模块
    print('二次创新模块_GSConv_Input size:', input.size())  # 打印输入张量的形状
    print('二次创新模块_GSConv_Output size:', output.size())  # 打印输出张量的形状
    print('即插即用模块限时特惠49.9，马上将恢复89.9原价。交流群小伙伴，免费享用二次创新模块!')
