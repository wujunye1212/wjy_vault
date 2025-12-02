import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C3

'''
来自IJCNN 2024论文 来自图像去雾任务的论文
即插即用模块：  MSPLCK多尺度特征提取模块  EPA 增强并行注意力模块    
              将这两个模块结合取一个新的名字：MSEPA 
              
MSPLCK模块的作用：
MSPLCK模块旨在解决图像去雾任务中的多尺度特性问题。具体作用包括：

多尺度特征提取：通过并行使用不同大小的扩张卷积核，MSPLCK模块能够同时捕获图像中的大尺度和小尺度特征。
            大卷积核关注全局特征，捕捉雾气浓度高的区域；小卷积核关注局部细节，恢复纹理信息。
大感受野：大核卷积的使用使得MSPLCK模块具有较大的感受野，类似于Transformer中的自注意力机制，能够处理长距离依赖关系，更好地去除图像中的雾气。
特征融合：不同尺度的特征在通道维度上进行拼接，然后通过多层感知机（MLP）进行融合，以综合不同尺度的信息，提高去雾效果。

EPA模块的作用：
EPA模块旨在处理图像中不均匀的雾气分布问题，并提高去雾网络的注意力机制效果。具体作用包括：

并行注意力机制：EPA模块并行使用了三种不同的注意力机制——简单像素注意力（SPA）、通道注意力（CA）和像素注意力（PA）。
             这种并行设计使得模块能够同时提取全局共享信息和位置依赖的局部信息，更有效地应对不均匀的雾气分布。
特征融合与增强：三种注意力机制的输出在通道维度上进行拼接，然后通过MLP进行融合，以增强特征表示。
             最终，融合后的特征与原始特征相加。
MSPLCK模块通过多尺度特征提取和大感受野设计，提高了网络对图像中不同尺度雾气特征的处理能力；
而EPA模块通过并行注意力机制，有效处理了图像中不均匀的雾气分布问题，增强了网络的注意力机制效果。
这两个模块的结合使得在去雾任务中取得了显著的性能提升。

MSEPA 模块适用于：图像去雾，目标检测，图像分割，语义分割，图像增强等所有计算机视觉CV任务通用模块

'''
class MSEPA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        # Simple Pixel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x  #用于残差连接
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x

# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 实例化模型对象
    model = MSEPA (dim=32)
    # 生成随机输入张量
    input = torch.randn(1, 32, 64, 64)
    # 执行前向传播
    output = model(input)
    print('input_size:',input.size())
    print('output_size:',output.size())

