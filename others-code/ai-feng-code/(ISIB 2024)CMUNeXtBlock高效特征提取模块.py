import torch
import torch.nn as nn
'''
CMUNeXtBlock 是 CMUNeXt 网络的核心模块，专注于高效提取全局上下文信息，同时保持轻量化和计算效率。
其主要作用包括：
1.全局信息提取：
使用大卷积核和深度卷积克服普通卷积的局部感受野限制，
增强网络对远距离特征的感知能力，适应医学图像中多尺度和复杂的结构。
2.减少计算开销：
使用深度可分离卷积将卷积分解为通道内卷积和点卷积，显著降低参数量和计算成本。
3.混合空间和通道信息：
利用反向瓶颈扩展中间特征层的维度，实现更充分的空间和通道特征融合。
4.提升训练稳定性与推理效率：通过残差连接和批归一化确保模型训练过程中的稳定性，并提高推理速度。
CMUNeXtBlock 通过大卷积核的深度卷积、反向瓶颈设计以及残差连接，在轻量化的同时高效提取全局和局部特征信息。
这使得它在医学图像分割任务中能够取得优异的表现，同时满足边缘设备部署的高效性需求。
适用于：医学图像分割，语义分割，实例分割，目标检测等所有CV2d任务通用的即插即用模块
'''
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out,kernel_size=3, stride=2, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out,kernel_size,stride)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out,kernel_size=3, stride=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.randn(1, 32, 64, 64)
    model = CMUNeXtBlock(ch_in=32,ch_out=64,kernel_size=3,stride=1)
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
