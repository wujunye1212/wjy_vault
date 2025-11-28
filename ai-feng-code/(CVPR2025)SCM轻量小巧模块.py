import torch
import torch.nn as nn
from functools import partial
import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

'''
来自CVPR 2025 顶会
即插即用模块： SCM 特征位移混合模块
带来两个二次创新模块 ： SPConv移动风车卷积模块  ； SCEU 移动有效上采样模块

主要内容：
模型二值化在实现卷积神经网络（CNN）的实时高效计算方面取得了显著进展，为视觉Transformer（ViT）在边缘设备上的部署挑战提供了潜在解决方案。
然而，由于CNN和Transformer架构的结构差异，直接将二值化CNN策略应用于ViT模型会导致性能显著下降。
为解决这一问题，我们提出了BHViT——一种适合二值化的混合ViT架构及其全二值化模型，其设计基于以下三个重要观察：

1.局部信息交互与分层特征聚合：BHViT利用从粗到细的分层特征聚合技术，减少因冗余token带来的计算开销。
2.基于移位操作的新型模块：提出一种基于移位操作的模块（SCM），在不显著增加计算负担的情况下提升二值化多层感知机（MLP）的性能。
3.量化分解的注意力矩阵二值化方法：提出一种基于量化分解的创新方法，用于评估二值化注意力矩阵中各token的重要性。

该Shift_channel_mix（SCM）模块是论文中提出的一个轻量化模块，用于增强二进制多层感知器（MLP）在二进制视觉变换器（BViT）中的表现。
它通过对输入特征图进行不同的移位操作，帮助缓解信息丢失和梯度消失的问题，从而提高网络的性能，同时避免增加过多的计算开销。
SCM模块的主要操作包括：
1.水平移位（Horizontal Shift）：通过torch.roll函数将特征图的列按指定的大小进行右/左移操作。这种操作模拟了在处理二进制向量时的特征循环，增强了表示能力。
2.垂直移位（Vertical Shift）：类似于水平移位，垂直移位会使特征图的行发生上下移动。这有助于捕获跨行的信息，同时适应不同的特征维度。
在代码实现中，torch.chunk将输入特征图沿着通道维度分成四个部分，之后通过不同的移位操作处理每一部分，最后将处理后的四个部分通过torch.cat拼接起来，形成最终的输出。

SCM模块适合：目标检测，图像分割，语义分割，图像增强，图像去噪，遥感语义分割，图像分类等所有CV任务通用的即插即用模块
这个SCM轻量小巧模块，建议最好搭配其它模块一起使用！

'''

class Shift_channel_mix(nn.Module):
    def __init__(self, shift_size=1):
        super(Shift_channel_mix, self).__init__()
        self.shift_size = shift_size

    def forward(self, x):  # x的张量 [B,C,H,W]
        x1, x2, x3, x4 = x.chunk(4, dim=1)

        x1 = torch.roll(x1, self.shift_size, dims=2)  # [:,:,1:,:]

        x2 = torch.roll(x2, -self.shift_size, dims=2)  # [:,:,:-1,:]

        x3 = torch.roll(x3, self.shift_size, dims=3)  # [:,:,:,1:]

        x4 = torch.roll(x4, -self.shift_size, dims=3)  # [:,:,:,:-1]

        x = torch.cat([x1, x2, x3, x4], 1)

        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
#二次创新模块 SPConv  移动风车形状卷积
class SPConv(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        # 定义4种非对称填充方式，用于风车形状卷积的实现
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]  # 每个元组表示 (左, 上, 右, 下) 填充
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]  # 创建4个填充层

        # 定义水平方向卷积操作，卷积核大小为 (1, k)，步幅为 s，输出通道数为 c2 // 4
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)

        # 定义垂直方向卷积操作，卷积核大小为 (k, 1)，步幅为 s，输出通道数为 c2 // 4
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)

        # 最终合并卷积结果的卷积层，卷积核大小为 (2, 2)，输出通道数为 c2
        self.cat = Conv(c2, c2, 2, s=1, p=0)

        self.shift_size = 1
    def forward(self, x):
        # 对输入 x 进行不同填充和卷积操作，得到四个方向的特征
        yw0 = self.cw(self.pad[0](x))  # 水平方向，第一个填充方式
        yw1 = self.cw(self.pad[1](x))  # 水平方向，第二个填充方式
        yh0 = self.ch(self.pad[2](x))  # 垂直方向，第一个填充方式
        yh1 = self.ch(self.pad[3](x))  # 垂直方向，第二个填充方式

        x1 = torch.roll(yw0, self.shift_size, dims=2)  # [:,:,1:,:]

        x2 = torch.roll(yw1, -self.shift_size, dims=2)  # [:,:,:-1,:]

        x3 = torch.roll(yh0, self.shift_size, dims=3)  # [:,:,:,1:]

        x4 = torch.roll(yh1, -self.shift_size, dims=3)  # [:,:,:,:-1]

        out = torch.cat([x1, x2, x3, x4], 1)
        # 将四个卷积结果在通道维度拼接，并通过一个额外的卷积层处理，最终输出
        return self.cat(out)  # 在通道维度拼接，并通过 cat 卷积层处理


#二次创新模块 SEUB 移动有效上采样模块

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class SCEU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(SCEU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.SCM = Shift_channel_mix(shift_size=1)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        # x = channel_shuffle(x, self.in_channels)
        x = self.SCM(x)
        x = self.pwc(x)
        return x

# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    input = torch.randn(1,32,64, 64)  # 创建一个形状为 (1,32,64, 64)
    SCM = Shift_channel_mix()
    output = SCM(input)  # 通过SCM模块计算输出
    print('SCM_Input size:', input.size())  # 打印输入张量的形状
    print('SCM_Output size:', output.size())  # 打印输出张量的形状

    input = torch.randn(1, 32, 64, 64)  # 创建一个形状为 (1,32,64, 64)
    SPConv = SPConv(32,32) #二次创新SPConv卷积模块
    output = SPConv(input)
    print('二次创新SPConv_Input size:', input.size())  # 打印输入张量的形状
    print('二次创新SPConv_Output size:', output.size())  # 打印输出张量的形状

    input = torch.randn(1, 32, 64, 64)  # 创建一个形状为 (1,32,64, 64)
    SCEU = SCEU(32,32) #二次创新SCEU上采样模块
    output = SCEU(input)
    print('二次创新SCEU_Input size:', input.size())  # 打印输入张量的形状
    print('二次创新SCEU_Output size:', output.size())  # 打印输出张量的形状