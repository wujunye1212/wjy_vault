import torch
import torch.nn as nn
import numpy as np
from numpy.random import RandomState

"""
    论文地址：https://www.sciencedirect.com/science/article/pii/S1361841524002743
    论文题目：Fourier Convolution Block with global receptive field for MRI reconstruction（2025 一区TOP）
    中文题目：用于 MRI 重建的具有全局感受野的傅里叶卷积模块（2025 一区TOP）
    讲解视频：https://www.bilibili.com/video/BV1YVekzVEDB/
        傅里叶卷积模块（Fourier Convolution Block, FCB）：
        实际意义：①传统 CNN“看不全”问题：MRI欠采样会产生“全局分布的伪影”，但普通CNN靠小卷积核（如3×3）只能关注局部，“看不全”全局，伪影消不干净。
                ②大感受野模型“又慢又难训”的问题：传统卷积对特征图所有通道进行相同操作，会产生大量冗余计算，特征部分通道信息重复，无需全通道密集计算。
        实现方式：①将图像特征从空间域到频域；
                ②再通过频域卷积公式，将频域特征实部和虚部进行计算，可以决定在频域里放大或抑制哪些成分，完成类似卷积效果。
                ③通过逆傅里叶变换变回空间域，得到特征图。
"""

def complexinit(weights_real, weights_imag, criterion):
    # 获取卷积核形状信息
    output_chs, input_chs, num_rows, num_cols = weights_real.shape
    fan_in = input_chs          # 输入通道数
    fan_out = output_chs        # 输出通道数

    # 初始化方式选择：Glorot（Xavier）或 He 初始化
    if criterion == 'glorot':
        s = 1. / np.sqrt(fan_in + fan_out) / 4.
    elif criterion == 'he':
        s = 1. / np.sqrt(fan_in) / 4.
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    # 随机数生成器
    rng = RandomState()
    kernel_shape = weights_real.shape

    # 生成复数权重的模和相位
    modulus = rng.rayleigh(scale=s, size=kernel_shape)       # 模长来自 Rayleigh 分布
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)  # 相位均匀分布

    # 将模长和相位转换为实部和虚部
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)

    # 将 numpy 转换为 torch.Tensor，并赋值给参数
    weights_real.data = torch.Tensor(weight_real)
    weights_imag.data = torch.Tensor(weight_imag)

class FCB(nn.Module):
    def __init__(self, input_chs: int, num_rows: int, num_cols: int, stride=1, init='he'):
        super(FCB, self).__init__()
        # 定义频域权重（实部和虚部），大小为 (1, 输入通道, H, W//2+1)
        self.weights_real = nn.Parameter(torch.Tensor(1, input_chs, num_rows, int(num_cols // 2 + 1)))
        self.weights_imag = nn.Parameter(torch.Tensor(1, input_chs, num_rows, int(num_cols // 2 + 1)))

        # 调用复数权重初始化函数
        complexinit(self.weights_real, self.weights_imag, init)

        # 保存输入图像的空间尺寸
        self.size = (num_rows, num_cols)
        self.stride = stride

    def forward(self, x):
        # 对输入张量做 2D 实数傅里叶变换 (频域表示)
        x = torch.fft.rfftn(x, dim=(-2, -1), norm=None)
        x_real, x_imag = x.real, x.imag   # 取出实部和虚部

        # 根据复数乘法的基本规则：
        # 频域卷积公式： (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        y_real = torch.mul(x_real, self.weights_real) - torch.mul(x_imag, self.weights_imag)
        y_imag = torch.mul(x_real, self.weights_imag) + torch.mul(x_imag, self.weights_real)

        # 将结果转换回时域（逆傅里叶变换）
        x = torch.fft.irfftn(torch.complex(y_real, y_imag), s=self.size, dim=(-2, -1), norm=None)

        # 如果 stride=2，则进行下采样（取每隔一个像素点）
        if self.stride == 2:
            x = x[..., ::2, ::2]

        return x

    def loadweight(self, ilayer):
        # 将已有的卷积层权重映射到频域
        weight = ilayer.weight.detach().clone()
        fft_shape = self.weights_real.shape[-2]

        # 翻转卷积核
        weight = torch.flip(weight, [-2, -1])

        # 填充至傅里叶变换所需大小
        pad = torch.nn.ConstantPad2d(
            padding=(0, fft_shape - weight.shape[-1], 0, fft_shape - weight.shape[-2]),
            value=0
        )
        weight = pad(weight)

        # 卷积核移位，使其中心对齐
        weight = torch.roll(weight, (-1, -1), dims=(-2, -1))

        # 做傅里叶变换，并调整通道顺序
        weight_kc = torch.fft.fftn(weight, dim=(-2, -1), norm=None).transpose(0, 1)
        # 只保留实数 FFT 的一半频谱
        weight_kc = weight_kc[..., :weight_kc.shape[-1] // 2 + 1]

        # 赋值到 FCB 模块的参数
        self.weights_real.data = weight_kc.real
        self.weights_imag.data = weight_kc.imag

if __name__ == "__main__":
    # 构造输入张量 (batch=1, 通道数=32, 高=50, 宽=50)
    x = torch.randn(1, 32, 50, 50)
    model = FCB(input_chs=32, num_rows=50, num_cols=50)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")