import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
# 论文翻译地址：https://blog.csdn.net/weixin_38594676/article/details/140274207

'''
I2U-Net：具有丰富信息交互的双路径 U-Net 用于医学图像分割
         SCI一区 2024
多功能信息交互即插即用模块:MFII

尽管U形网络在许多医学图像分割任务中取得了显著的性能，但它们很少建模层次化之间的顺序关系。
这一弱点使得当前层难以有效利用前一层的历史信息，导致对具有模糊边界和不规则形状病变的分割结果不好。
为了解决这一问题，我们提出了一种新颖的双路径 U-Net，称为 I2U-Net。

新提出的网络通过双路径之间的丰富信息交互，鼓励历史信息的重复使用和重新探索，
使得深层可以学习更全面的特征，既包含低层次的详细描述，又包含高层次的语义抽象。
具体来说，我们引入了一个多功能信息交互模块（MFII），它通过统一设计可以建模跨路径、跨层次和跨路径-层次的信息交互，
使得所提出的 I2U-Net 表现类似于展开的 RNN，并享有建模时间序列信息的优势。

简单介绍一下MFII模块：
I2U-Net是一个具有丰富信息交互的双路径U-Net。
其中一条路径使用医学图像作为输入x，提取像传统U-Net一样的图像特征信息。
相比之下，另一条路径使用零初始化的可学习矩阵作为输入h，在深度上使用共享的卷积核存储隐藏状态信息。
这种结构使得I2U-Net可以类似于展开的RNN工作，并享受其优势，包括建模层次化层之间的时间序列关系，充分利用历史信息。

适用于：图像分割，目标检测，图像分类等所有计算机视觉CV2维任务通用即插即用模块

'''
class MFII(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, stride=1, downsample=None,
                 rla_channel=32, SE=False, ECA_size=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(MFII, self).__init__()
        planes =inplanes
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # if groups != 1 or base_width != 64:
        # raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes +rla_channel, planes, stride=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.conv3 = conv1x1(planes,outplanes)
        self.conv4 = conv1x1(planes, rla_channel)
        self.bnh = norm_layer(rla_channel)

        self.averagePooling = None
        if downsample is not None and stride != 1:
            self.averagePooling = nn.AvgPool2d((2, 2), stride=(2, 2))

        self.se = None
        if SE:
            self.se = SELayer(planes * self.expansion, reduction)

        self.eca = None
        if ECA_size != None:
            self.eca = eca_layer(planes * self.expansion, int(ECA_size))


    def forward(self, x, h):
        identity = x
        x = torch.cat((x, h), dim=1)   # [8, 96, 56, 56]

        out = self.conv1(x)            # [8, 64, 56, 56]
        out = self.bn1(out)            # [8, 64, 56, 56]
        out = self.relu(out)

        out = self.conv2(out)          # [8, 64, 56, 56]
        out = self.bn2(out)
        if self.se != None:
            out = self.se(out)
        if self.eca != None:
            out = self.eca(out)
        y1 = out
        y = y1+identity
        y = self.conv3(y)
        h1 = self.conv4(y1)
        h = self.relu(self.bnh(h+h1))
        if self.averagePooling is not None:
            h = self.averagePooling(h)
            y = self.averagePooling(y)
        return y, h

if __name__ == '__main__':
    # 定义输入数据的形状
    batch_size = 8
    in_channels = 64
    h_channels = 32
    height, width = 56, 56
    # 创建模型
    model = MFII(inplanes=in_channels, outplanes=128, stride=2,downsample=True, rla_channel=h_channels, SE=True, ECA_size=3)
    # 创建虚拟输入数据
    x = torch.randn(batch_size, in_channels, height, width)
    h = torch.zeros(batch_size, h_channels, height, width)
    # print("h 张量:",h )
    # 输出结果
    print("x input:", x.shape)
    print("h input:", h.shape)

    # 前向传播
    x, h = model(x, h)
    # 输出结果
    print("x output:", x.shape)
    print("h output:", h.shape)
