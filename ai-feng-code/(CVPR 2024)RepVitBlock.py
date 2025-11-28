import torch.nn as nn
from timm.models.layers import SqueezeExcite
import torch
'''
                 CVPR 2024 顶会
即插即用模块：RepViTBlock
最近，与轻量级卷积神经网络 （CNN） 相比，轻量级 Vision Transformer （ViTs） 
在资源受限的移动设备上表现出卓越的性能和更低的延迟。研究人员发现了轻量级 ViT 和轻量级 CNN 之间的许多结构联系。
然而，它们之间在块结构、宏观和微观设计方面的显着架构差异尚未得到充分检查。在这项研究中，
我们从 ViT 的角度重新审视了轻量级 CNN 的高效设计，并强调了它们在移动设备中的前景。
具体来说，我们通过集成轻量级 ViT 的高效架构设计，逐步增强了标准轻量级 CNN（即 MobileNetV3）的移动友好性。
这最终产生了一个新的纯轻量级 CNN 系列，即 RepViT。广泛的实验表明，RepViT 的性能优于现有的最先进的轻量级 ViT，
并且在各种视觉任务中表现出良好的延迟。

RepViTBlock 模块的作用总结如下：
1.分离 Token Mixer 和 Channel Mixer：
RepViTBlock 模块将 token mixer（通过 3x3 深度卷积处理空间信息）和 channel mixer（通过 1x1 卷积处理通道信息）分离开来。
这种设计提高了空间信息与通道信息的处理效率，受到了 Vision Transformer (ViT) 架构的启发。
2.结构重参数化：
RepViTBlock 在训练阶段使用结构重参数化技术，通过多分支结构增强模型的学习能力，而在推理阶段简化网络结构，减少计算和内存开销。
这种技术特别适合在移动设备上应用，有效降低了延迟和资源消耗。
3.提升移动设备效率：
RepViTBlock 能在保持高准确率的同时，大幅度提升模型在移动设备上的推理效率，
适用于图像分类、目标检测、实例分割、语义分割等所有计算机视觉CV任务
'''

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv
class RepViTBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2,hidden_dim=80, use_se=True, use_hs=True):
        super(RepViTBlock, self).__init__()
        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input = torch.randn(1, 32, 64, 64)
    # 创建一个 RepViTBlock 模块实例
    model = RepViTBlock(inp=32,oup=32,kernel_size=3, stride=2)
    # 执行前向传播
    output = model(input)
    # 打印输入和输出的形状
    print('input_size:',input.size())
    print('output_size:',output.size())
