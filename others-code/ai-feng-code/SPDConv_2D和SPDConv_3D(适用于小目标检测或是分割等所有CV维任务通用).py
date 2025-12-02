import torch
import torch.nn as nn

'''
无卷积步长或池化:用于低分辨率图像和小目标物体的新 CNN 模块 SPD-Conv
即插即用下采样模块：SPDConv

卷积神经网络(CNNs)在图像分类和目标检测等计算机视觉任务中取得了显著的成功。
然而，当图像分辨率较低或物体较小时，它们的性能会迅速下降。在本文中，
我们指出这根源在于现有CNN设计体系结构中存在缺陷，即使用卷积步长或池化层，
这导致了细粒度信息的丢失和低效特征表示的学习。

为此，我们提出了一个名为SPD-Conv的新的CNN构建块来代替每个卷积步长和每个池化层(因此完全消除了它们存在缺陷)。
并通过实验证明，特别是在处理低分辨率图像和小目标物体等任务时效果不错。

适用于：小目标检测、小目标分割等所有CV2维任务通用的下采样卷积模块
'''

class SPDConv_2D(nn.Module):
    def __init__(self, c1, c2, k=1):
        super().__init__()
        c1 = c1 * 4
        self.conv = nn.Conv2d(c1, c2, k)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()
    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.act(self.bn(self.conv(x)))
class SPDConv_3D(nn.Module):

    def __init__(self, c1, c2, k=1):
        super().__init__()
        c1 = c1 * 8  # For 3D version, concatenate 8 blocks
        self.conv = nn.Conv3d(c1, c2, k)
        self.bn = nn.BatchNorm3d(c2)
        self.act = nn.ReLU()

    def forward(self, x):
        # 3D version: concatenate along the depth, height, and width dimensions
        x = torch.cat([x[..., ::2, ::2, ::2], x[..., 1::2, ::2, ::2],
                       x[..., ::2, 1::2, ::2], x[..., 1::2, 1::2, ::2],
                       x[..., ::2, ::2, 1::2], x[..., 1::2, ::2, 1::2],
                       x[..., ::2, 1::2, 1::2], x[..., 1::2, 1::2, 1::2]], 1)
        return self.act(self.bn(self.conv(x)))

# SPDConv3D 模块
if __name__ == '__main__':
    block = SPDConv_3D(c1=32, c2=32)  # 3D任务
    input = torch.randn(1, 32, 16, 16, 16)  # 输入为 3D 张量：N C D H W
    output = block(input)
    print('3D_Input shape:', input.shape)
    print('3D_Output shape:', output.shape)


    input = torch.randn(1, 64, 32, 32)
    SPDConv = SPDConv_2D(c1=64, c2=64)  # 2D任务
    output = SPDConv(input)
    print('2D_input_size:', input.size())
    print('2D_output_size:', output.size())




