import torch
import torch.nn as nn
from timm.models.layers import DropPath
BNNorm2d = nn.BatchNorm2d
LNNorm = nn.LayerNorm
Activation = nn.GELU
#CVPR 2025 医学图像分割
##看Ai缝合怪b站视频：2025.6.23更新的视频
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            BNNorm2d(ch_out),
            Activation()
        )

    def forward(self, x):
        x = self.up(x)
        return x

class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=False),
            BNNorm2d(ch_out),
            Activation()
        )

    def forward(self, x):
        x = self.down(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, groups=1):
        super(ResBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=1, padding=1)
        self.bn1 = BNNorm2d(planes)
        self.act = Activation()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, groups=groups, padding=1)
        self.bn2 = BNNorm2d(planes)

        if self.inplanes != self.planes:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=1),
                BNNorm2d(planes)
            )
    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)

        if self.inplanes != self.planes:
            identity = self.down(x)

        out = self.bn2(out) + identity
        out = self.act(out)

        return out


class LSB_down_or_up(nn.Module):
    def __init__(self, inplanes,  outplanes,down_or_up=None, groups=1):
        super(LSB_down_or_up, self).__init__()
        hidden_planes= 2*inplanes
        if down_or_up is None:
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                # ResBlock(inplanes=planes, planes=planes, groups=groups),
            )

        elif down_or_up == 'down':
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                # ResBlock(inplanes=planes, planes=planes, groups=groups),
                down_conv(hidden_planes, outplanes)
            )
        elif down_or_up == 'up':
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                # ResBlock(inplanes=planes, planes=planes, groups=groups),
                up_conv(hidden_planes, outplanes),
            )

    def forward(self, x):
        out = self.BasicBlock(x)
        return out


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = Activation()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GSB(nn.Module):
    def __init__(self, in_dim, out_dim,  pool_size=3, mlp_ratio=4., drop=0., drop_path=0., sr_ratio=1):
        super().__init__()

        self.in_dim = in_dim
        self.dim = out_dim

        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=3,  padding=1)
        self.norm1 = GroupNorm(out_dim)
        self.attn = Pooling(pool_size=pool_size) #
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = GroupNorm(out_dim)
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = Mlp(in_features=out_dim, hidden_features=mlp_hidden_dim, out_features=out_dim, drop=drop)

    def forward(self, x):
        x = self.proj(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input= torch.randn(1, 32, 64, 64)
    # 创建 GSB 模块
    GSB = GSB(in_dim=32,out_dim=32)
    # 将输入图像传入GSB 模块进行处理
    output = GSB(input)
    # 输出结果的形状
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新-GSB_input_size:', input.size())
    print('Ai缝合即插即用模块永久更新-GSB_output_size:', output.size())

    # 创建 LSB 模块
    LSB_up = LSB_down_or_up(inplanes=32,outplanes=32,down_or_up='up')
    # 将输入图像传入LSB_up 模块进行上采样处理
    output = LSB_up(input)
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新-LSB_up_input_size:', input.size())
    print('Ai缝合即插即用模块永久更新-LSB_up_output_size:', output.size())

    # 创建 LSB 模块
    LSB_down = LSB_down_or_up(inplanes=32,outplanes=32,down_or_up='down')
    # 将输入图像传入LSB_down模块进行下采样处理
    output = LSB_down(input)
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新-LSB_down_input_size:', input.size())
    print('Ai缝合即插即用模块永久更新-LSB_down_output_size:', output.size())