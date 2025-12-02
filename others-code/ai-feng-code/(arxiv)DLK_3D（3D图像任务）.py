import torch
import torch.nn as nn

from timm.models.layers import DropPath
# 论文：D-Net：具有动态特征融合的动态大核，用于体积医学图像分割（3D图像任务）
# https://arxiv.org/abs/2403.10674
#代码：https://github.com/sotiraslab/DLK
'''
层次化的变压器（transformer）模型在医学图像分割领域取得了显著的成功，
这归功于它们拥有广阔的接收域以及有效利用全局长程上下文信息的能力。
卷积神经网络（CNNs）同样可以通过使用大尺寸的卷积核来实现大的接收域，
使它们能够在较少的模型参数下达到竞争性的性能。
然而，使用大卷积核的CNNs在适应性捕捉具有形状和
大小大幅变化的器官的多尺度特征方面仍受到限制，
这是由于固定大小的卷积核的使用。
此外，它们无法高效地利用全局上下文信息。
为了解决这些局限性，我们提出了动态大核（DLK）模块。
DLK模块采用多个具有不同大小和膨胀率的大核来捕捉多尺度特征。
随后，使用一种动态选择机制，根据全局信息自适应地突出最重要的空间特征。
此外，DFF模块被提出，根据其全局信息自适应地融合多尺度局部特征图。
'''
class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        drop = 0.
        self.fc1 = nn.Conv3d(dim, dim * 4, 1)
        self.dwconv = nn.Conv3d(dim * 4, dim * 4, 3, 1, 1, bias=True, groups=dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(dim * 4, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DLK(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att_conv1 = nn.Conv3d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.att_conv2 = nn.Conv3d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)

        self.spatial_se = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        att1 = self.att_conv1(x)
        att2 = self.att_conv2(att1)

        att = torch.cat([att1, att2], dim=1)
        avg_att = torch.mean(att, dim=1, keepdim=True)
        max_att, _ = torch.max(att, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        output = att1 * att[:, 0, :, :, :].unsqueeze(1) + att2 * att[:, 1, :, :, :].unsqueeze(1)
        output = output + x
        return output
class DLKModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_1 = nn.Conv3d(dim, dim, 1)
        self.act = nn.GELU()
        self.spatial_gating_unit = DLK(dim)
        self.proj_2 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x

class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv3d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False)

        self.conv1 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)

        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        output = output * att
        return output

if __name__ == '__main__':
    input = torch.randn(1, 32, 16, 64, 64) # x: (B, C, D,H, W) 3D图像维度
    model = DLKModule(32)
    output = model(input)
    print("DLKModule_input size:", input.size())
    print("DLKModule_Output size:", output.size())