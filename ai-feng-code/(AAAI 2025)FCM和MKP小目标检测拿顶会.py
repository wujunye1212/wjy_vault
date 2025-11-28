import torch
import torch.nn as nn


#论文： https://arxiv.org/pdf/2504.20670
#务必看我b站：2025.05.14的视频讲解

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

class Channel_aifengheguai(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = self.dconv = nn.Conv2d(
            dim, dim, 3,
            1, 1, groups=dim
        )
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)

        return x6
class Spatial_aifengheguai(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)

        return x6

class FCM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1_aifhg = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12_aifhg = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123_aifhg = Conv(dim - dim // 4, dim, 1, 1)
        self.conv2_aifhg = Conv(dim // 4, dim, 1, 1)
        self.spatial_aifhg = Spatial_aifengheguai(dim)
        self.channel_aifhg = Channel_aifengheguai(dim)
    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1_aifhg(x1)  #K=3
        x3 = self.conv12_aifhg(x3) #K=3
        x3 = self.conv123_aifhg(x3) #K=1
        x4 = self.conv2_aifhg(x2) #K=1
        x33 = self.spatial_aifhg(x4) * x3
        x44 = self.channel_aifhg(x3) * x4
        x5 = x33 + x44
        return x5

class MKPConv(nn.Module):
    def __init__(self, dim, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv1_aifhg = nn.Conv2d(
            dim, dim, 3,
            1, 1, groups=dim
        )
        self.conv2_aifhg = Conv(dim, dim, k=1, s=1, )
        self.conv3_aifhg = nn.Conv2d(
            dim, dim, 5,
            1, 2, groups=dim
        )
        self.conv4_aifhg = Conv(dim, dim, 1, 1)
        self.conv5_aifhg = nn.Conv2d(
            dim, dim, 7,
            1, 3, groups=dim
        )
    def forward(self, x):
        x1 = self.conv1_aifhg(x)
        x2 = self.conv2_aifhg(x1)
        x3 = self.conv3_aifhg(x2)
        x4 = self.conv4_aifhg(x3)
        x5 = self.conv5_aifhg(x4)
        x6 = x5 + x
        return x6

# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    input = torch.randn(1,32,64, 64)  # 创建一个形状为 (1,32,64, 64)
    FCM = FCM(32)
    output = FCM(input)  # 通过FCM模块计算输出
    print('\n Ai缝合怪永久更新-FCM_Input size:', input.size())  # 打印输入张量的形状
    print('Ai缝合怪永久更新-FCM_Output size:', output.size())  # 打印输出张量的形状

    input = torch.randn(1, 32, 64, 64)  # 创建一个形状为 (1,32,64, 64)
    MKPConv = MKPConv(32)
    output = MKPConv(input)  # 通过MKPConv模块计算输出
    print('Ai缝合怪永久更新-MKPConv_Input size:', input.size())  # 打印输入张量的形状
    print('Ai缝合怪永久更新-MKPConv_Output size:', output.size())  # 打印输出张量的形状