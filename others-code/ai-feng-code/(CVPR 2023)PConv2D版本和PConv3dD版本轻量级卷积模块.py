import torch
from torch import Tensor, nn

# https://github.com/JierunChen/FasterNet/tree/master
# https://arxiv.org/abs/2303.03667

'''
                     CVPR 2023
即插即用模块：PConv轻量级卷积模块

为了设计快速神经网络，许多工作一直专注于减少浮点运算 （FLOP） 的数量。
然而，我们观察到 FLOP 的这种减少不一定会导致类似程度的延迟降低。
这主要源于效率低下的每秒浮点运算数 （FLOPS）。
为了实现更快的网络，我们重新审视了流行的算子，并证明如此低的 FLOPS 主要是由于算子频繁的内存访问，
尤其是深度卷积。因此，我们提出了一种新的部分卷积 （PConv），
它通过同时减少冗余计算和内存访问来更有效地提取空间特征。

在我们的 PConv 的基础上，我们进一步提出了 FasterNet，这是一个新的神经网络系列，
在各种设备上都能获得比其他设备高得多的运行速度，而不会影响各种视觉任务的准确性。
例如，在 ImageNet1k 上，我们的微型 FasterNet-T0 在 GPU、CPU 和 ARM 处理器上
分别比 MobileViT-XXS 快 2.8×、3.3× 和 2.4× 倍，同时准确率高 2.9%。

我们的部分卷积 （PConv） 通过仅在少数输入通道上应用滤波器，
而保持其余通道不变，从而快速高效。
适用于：对模型执行速度有要求的任务，图像分类任务等所有cv2维任务通用模块

'''
class PConv2D(nn.Module):
    def __init__(self, dim, n_div, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x
    def forward(self, x):
        return self.forward(x)



class PConv3D(nn.Module):
    def __init__(self, dim, n_div, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv3d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

    def forward(self, x):
        return self.forward(x)


#  PConv2D模块
if __name__ == '__main__':
# 创建一个2D输入张量，形状为 B C H W
     input = torch.randn(1, 32, 64, 64)
     model = PConv2D(dim=32, n_div=4)
# 创建一个3D输入张量，形状为 B C D H W
#      input = torch.randn(1, 32, 16, 64, 64)
#      model = PConv3D(dim=32, n_div=4)
#    前向传播，获取输出
     output = model(input)
     print(f"Input shape: {input.shape}")
     print(f"Output shape: {output.shape}")



