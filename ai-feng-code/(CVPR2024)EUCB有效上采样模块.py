import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

# Other types of layers can go here (e.g., nn.Linear, etc.)
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

class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(EUCB, self).__init__()

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
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x
class MEEM(nn.Module):
    def __init__(self, in_dim, out_dim, width=4, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        hidden_dim = in_dim // 2
        self.out = out_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, out_dim, 1, bias=False),
            norm(out_dim),
            act()
        )

    def forward(self, x):
        mid = self.in_conv(x)
        out = mid
        # print(out.shape)

        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)

        out = self.out_conv(out)

        return out


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge
# 二次创新模块 ESUM 有效协调上采样模块 冲csi一区
'''
ESUM 有效协调上采样模块 内容介绍：



'''
class MEUM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(MEUM, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.FE = MEEM(in_dim=in_channels,out_dim=out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.FE(x)
        return x
if __name__ == '__main__':
    EUCB = EUCB(32, 64) #输入通道数是32，输出通道数是64
    input = torch.randn(1,32, 64,64)
    output = EUCB(input)
    print(f"EUCB_输入张量的形状: {input.size()}")
    print(f"EUCB_输入张量的形状: {output.size()}")
    ESUM = MEUM(32, 64)  #输入通道数是32，输出通道数是64
    output = ESUM(input)
    print(f"二次创新模块_ESUM_输入张量的形状: {input.size()}")
    print(f"二次创新模块_ESUM_输入张量的形状: {output.size()}")

