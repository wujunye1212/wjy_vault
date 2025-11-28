import torch
import torch.nn as nn

from timm.models.layers import DropPath
# 论文：D-Net：具有动态特征融合的动态大核，用于体积医学图像分割（3D图像任务）
# https://arxiv.org/abs/2403.10674
#代码：https://github.com/sotiraslab/DLK
'''
动态特征融合（DFF）模块:
我们提出了一个动态特征融合（DFF）模块，基于全局信息自适应地融合多尺度局部特征（图2）。
它是通过根据其全局信息动态选择重要特征来实现的融合。
具体来说，特征映射F l 1和F l 2沿着通道连接起来。
为了确保以下块可以采用融合特性，
需要一个通道减少机制来将通道的数量减少到原来的通道。
DFF中的信道减少采用1×1×1卷积，而不是简单的全局×信息。
通过级联平均池化（AVGPool）、卷积层（Conv1）和Sigmoid激活来提取这些信息来描述特征的重要性。
主要用于3D医学图像分割任务,同时也适用于所有CV3D图像任务。
'''
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
    input1 = torch.randn(1, 32, 16, 64, 64) # x: (B, C, D,H, W) 3D图像维度
    input2 = torch.randn(1, 32, 16, 64, 64)  # x: (B, C, D,H, W) 3D图像维度
    model = DFF(32)
    output = model(input1,input2)
    print("DFF_input size:", input1.size())
    print("DFF_Output size:", output.size())