import torch
import torch.nn.functional as F
from torch import nn

"""
    论文地址：https://arxiv.org/abs/2304.08069
    论文题目：DETRs Beat YOLOs on Real-time Object Detection(CVPR 2024)
    中文题目：DETR 在实时目标检测方面击败 YOLO (CVPR 2024)
    讲解视频：https://www.bilibili.com/video/BV193zyYDEYD/
    基于卷积神经网络的跨尺度特征融合（CNN-based Cross-scale Feature Fusion, CCFF）：
        作用：在融合路径中通过多个卷积层融合块来实现跨尺度特征融合。
        结构组成：两个1×1的卷积层用于调整通道数，N个RepConv用于特征融合，通过逐元素相加的方式进行融合。
"""

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        # 定义卷积层，设置输入通道、输出通道、核大小、步幅、填充和偏置
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        # 定义批归一化层，适用于输出通道数
        self.norm = nn.BatchNorm2d(ch_out)
        # 定义激活函数，这里使用ReLU
        self.act = nn.ReLU()

    def forward(self, x):
        # 前向传播：卷积 -> 归一化 -> 激活
        return self.act(self.norm(self.conv(x)))

class RepVggBlock(nn.Module):
    # https://arxiv.org/pdf/2101.03697.pdf
    # 重参数化
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        # 定义3x3卷积层，步幅为1，填充为1
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        # 定义1x1卷积层，步幅为1，无填充
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        # 获取激活函数，这里使用ReLU
        self.act = nn.ReLU()

    def forward(self, x):
        # 前向传播：两个卷积的结果相加
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self):
        # 转换为部署模式
        if not hasattr(self, 'conv'):
            # 定义新的3x3卷积层用于部署
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        # 获取等效的卷积核和偏置
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        # 获取等效的卷积核和偏置
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        # 返回合并后的卷积核和偏置
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        # 将1x1卷积核填充为3x3
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        # 融合批归一化参数
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        # 返回融合后的卷积核和偏置
        return kernel * t, beta - running_mean * gamma / std

class CCFF(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CCFF, self).__init__()
        hidden_channels = int(out_channels * expansion)
        # 定义第一个卷积层
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)

        # 定义瓶颈层，包含多个RepVggBlock块
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])

        # 定义第二个卷积层
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)

        # 定义第三个卷积层或Identity
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        # 前向传播：特征融合
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)

# 输入 N C H W, 输出 N C H W
if __name__ == '__main__':
    # 实例化模型对象
    model = CCFF(in_channels=64, out_channels=64)

    # model = CCFF(in_channels=64, out_channels=128)

    # 创建随机输入张量
    input = torch.randn(1, 64, 32, 32)

    # 通过模型进行前向传播
    output = model(input)

    # 打印输入和输出的大小
    print('input_size:', input.size())
    print('output_size:', output.size())

    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息
