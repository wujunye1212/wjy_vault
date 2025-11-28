import math
import torch.nn as nn
import torch

'''
SCI 2024论文
注意力机制逐渐成为增强卷积神经网络 （CNN） 表示能力的必要条件。
尽管最近注意力机制研究取得了进展，但仍存在一些悬而未决的问题。
大多数现有方法都忽略了对多尺度特征表示、结构信息和远程通道依赖性进行建模，
这对于提供更具辨别性的注意力图至关重要。
本研究针对各种网络和数据集提出了一种新颖的、低开销的、高性能的注意力机制，具有很强的泛化能力。
这种机制被称为多尺度空间金字塔注意力 （MSPA），可用于解决其他注意力方法的局限性。
对于 MSPA 的关键组件，我们不仅开发了分层幻影卷积 （HPC） 模块，
该模块可以利用分层残差类连接在更细粒度的层面上提取多尺度空间信息，
还设计了空间金字塔重新校准 （SPR） 模块，该模块可以将结构正则化和结构信息集成到自适应组合机制中，
同时采用 Softmax 操作构建长距离通道依赖性。 MSPA可以作为即插即用模块方便地嵌入到各种 CNN 中。

MSPA模块的作用可以概括如下三点：
1.多尺度空间信息提取：MSPA通过一个称为HPC（分层幻象卷积）的模块来有效地提取多尺度的空间信息。
HPC模块首先将输入特征图沿通道维度分成多个子集，然后在每个子集上应用层次化残差卷积，
从而在不同尺度上进行特征增强。这种设计能够在更细粒度的层次上增强多尺度特征表示能力。

2.跨通道相互作用的建模：MSPA包含一个称为SPR（空间金字塔重校准）的模块，
通过适应性地聚合全局和局部特征响应，实现跨维度的相互作用。SPR模块同时使用1x1卷积来学习通道之间的关系，
从而在特征中保留更多结构信息和正则化效果，提升通道注意力的表现。

3.长距离通道依赖的建立：通过在SPR模块输出的通道注意力权重上应用Softmax操作，
MSPA能够在不同通道之间建立长距离的依赖关系。
这一步使得通道权重得以在局部和全局通道注意力之间进行自适应选择，进一步增强模型的特征表达能力。

适用于：目标检测，图像增强，图像分割，图像分类等所有计算机视觉CV任务通用模块
'''
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def convdilated(in_planes, out_planes, kSize=3, stride=1, dilation=1):
    """3x3 convolution with dilation"""
    padding = int((kSize - 1) / 2) * dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=kSize, stride=stride, padding=padding,
                     dilation=dilation, bias=False)

class SPRModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SPRModule, self).__init__()

        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)

        self.fc1 = nn.Conv2d(channels * 5, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out1 = self.avg_pool1(x).view(x.size(0), -1, 1, 1)
        out2 = self.avg_pool2(x).view(x.size(0), -1, 1, 1)
        out = torch.cat((out1, out2), 1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight
class MSPAModule(nn.Module):
    def __init__(self, inplanes, scale=3, stride=1, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality.
            scale: number of scale.
            stride: conv stride.
            stype: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(MSPAModule, self).__init__()

        self.width = inplanes
        self.nums = scale
        self.stride = stride
        assert stype in ['stage', 'normal'], 'One of these is suppported (stage or normal)'
        self.stype = stype

        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])

        for i in range(self.nums):
            if self.stype == 'stage' and self.stride != 1:
                self.convs.append(convdilated(self.width, self.width, stride=stride, dilation=int(i + 1)))
            else:
                self.convs.append(conv3x3(self.width, self.width, stride))

            self.bns.append(nn.BatchNorm2d(self.width))

        self.attention = SPRModule(self.width)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]

        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0 or (self.stype == 'stage' and self.stride != 1):
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](sp)

            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        feats = out
        feats = feats.view(batch_size, self.nums, self.width, feats.shape[2], feats.shape[3])

        sp_inp = torch.split(out, self.width, 1)

        attn_weight = []
        for inp in sp_inp:
            attn_weight.append(self.attention(inp))

        attn_weight = torch.cat(attn_weight, dim=1)
        attn_vectors = attn_weight.view(batch_size, self.nums, self.width, 1, 1)
        attn_vectors = self.softmax(attn_vectors)
        feats_weight = feats * attn_vectors

        for i in range(self.nums):
            x_attn_weight = feats_weight[:, i, :, :, :]
            if i == 0:
                out = x_attn_weight
            else:
                out = torch.cat((out, x_attn_weight), 1)

        return out


class MSPABlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes,baseWidth=30, scale=3,norm_layer=None, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality.
            planes: output channel dimensionality.
            stride: conv stride.
            downsample: None when stride = 1.
            baseWidth: basic width of conv3x3.
            scale: number of scale.
            norm_layer: regularization layer.
            stype: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(MSPABlock, self).__init__()
        planes=inplanes
        stride = 1,
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(math.floor(planes * (baseWidth / 64.0)))

        self.conv1 = conv1x1(inplanes, width * scale)
        self.bn1 = norm_layer(width * scale)

        self.conv2 = MSPAModule(width, scale=scale, stride = stride, stype=stype)
        self.bn2 = norm_layer(width * scale)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input = torch.randn(1, 64, 32, 32)
    # 1.创建一个 MSPABlock 模块实例
    MSPABlock= MSPABlock(inplanes=64)
    # 执行前向传播
    output = MSPABlock(input)
    # 打印输入和输出的形状
    print('MSPABlock_input_size:',input.size())
    print('MSPABlock_output_size:',output.size())

    # 2.创建一个 MSPA 模块实例

    MSPA = MSPAModule(inplanes=16,scale=4)
    # 执行前向传播
    output = MSPA(input)
    # 打印输入和输出的形状
    print('MSPA_input_size:', input.size())
    print('MSPA_output_size:', output.size())
