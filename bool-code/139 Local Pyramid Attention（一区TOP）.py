import torch
import torch.nn as nn

"""
    论文地址：https://ieeexplore.ieee.org/document/9895210
    论文题目：SwinPA-Net: Swin Transformer-Based Multiscale Feature Pyramid Aggregation Network for Medical Image Segmentation（一区 TOP）
    中文题目：SwinPA-Net：基于Swin-Transformer的多尺度特征金字塔聚合网络的图像分割（一区 TOP）
    讲解视频：https://www.bilibili.com/video/BV1geRBYiEp2/
    局部金字塔注意力（Local Pyramid Attention, LPA）：
        实际意义：①差异大与颜色相近问题：医学图像中病变类型差异巨大，且病变与周围组织颜色相近，这使得网络难以精准识别病变位置，传统的神经网络很难区分它们。
                ②不同大小病变结构特征困难：小的病变可能在特征图中仅占据很少像素，容易被忽略；大的病变则可能包含更多细节和变化。
        实现方式：①LPA 模块通过引入注意力机制，能引导网络聚焦于目标区域，增强对病变区域的关注，抑制无关信息，从而提高对病变位置的识别能力。
                ②LPA 模块利用金字塔结构，将特征图划分为不同大小和维度，分别获取局部和全局注意力图并融合。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】   
"""

# 定义通道注意力模块
class ChannelAttention(nn.Module):
    # 初始化函数，in_planes 表示输入通道数
    def __init__(self, in_planes):
        # 调用父类的初始化方法
        super(ChannelAttention, self).__init__()
        # 定义自适应平均池化层，将输入特征图池化为 1x1 的大小
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义自适应最大池化层，将输入特征图池化为 1x1 的大小
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 定义第一个卷积层，将输入通道数缩小为原来的 1/8
        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        # 定义 ReLU 激活函数
        self.relu1 = nn.ReLU()
        # 定义第二个卷积层，将通道数恢复为输入通道数
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        # 定义 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播函数
    def forward(self, x):
        # 对输入进行自适应平均池化，然后通过两层卷积和激活函数处理
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 对输入进行自适应最大池化，然后通过两层卷积和激活函数处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 将平均池化和最大池化的结果相加
        out = avg_out + max_out
        # 对相加的结果应用 Sigmoid 激活函数
        return self.sigmoid(out)


# 定义空间注意力模块
class SpatialAttention(nn.Module):
    # 初始化函数，kernel_size 表示卷积核大小，默认为 3
    def __init__(self, kernel_size=3):
        # 调用父类的初始化方法
        super(SpatialAttention, self).__init__()

        # 断言卷积核大小必须为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 根据卷积核大小设置填充值
        padding = 3 if kernel_size == 7 else 1

        # 定义卷积层，输入通道数为 2，输出通道数为 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        # 定义 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播函数
    def forward(self, x):
        # 沿通道维度计算平均池化结果
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 沿通道维度计算最大池化结果
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 将平均池化和最大池化的结果在通道维度上拼接
        x = torch.cat([avg_out, max_out], dim=1)
        # 通过卷积层处理拼接后的结果
        x = self.conv1(x)
        # 对卷积结果应用 Sigmoid 激活函数
        return self.sigmoid(x)

class LPA(nn.Module):
    # 初始化函数，in_channel 表示输入通道数
    def __init__(self, in_channel):
        # 调用父类的初始化方法
        super(LPA, self).__init__()
        # 实例化通道注意力模块
        self.ca = ChannelAttention(in_channel)
        # 实例化空间注意力模块
        self.sa = SpatialAttention()

    def forward(self, x):
        # --------------图上半部分--------------
        # 对原始输入应用通道注意力机制并相乘
        x4 = self.ca(x) * x
        # 对应用通道注意力后的结果应用空间注意力机制并相乘
        x4 = self.sa(x4) * x4

        # --------------图下半部分--------------
        # 沿第 2 个维度将输入特征图分成两部分
        x0, x1 = x.chunk(2, dim=2) # H
        # 对 x0 沿第 3 个维度再分成两部分
        x0 = x0.chunk(2, dim=3) # W
        # 对 x1 沿第 3 个维度再分成两部分
        x1 = x1.chunk(2, dim=3)
        # 对 x0 的两部分分别应用通道注意力机制并相乘
        x0 = [self.ca(x0[-2]) * x0[-2], self.ca(x0[-1]) * x0[-1]]
        # 对应用通道注意力后的 x0 两部分分别应用空间注意力机制并相乘
        x0 = [self.sa(x0[-2]) * x0[-2], self.sa(x0[-1]) * x0[-1]]
        # 对 x1 的两部分分别应用通道注意力机制并相乘
        x1 = [self.ca(x1[-2]) * x1[-2], self.ca(x1[-1]) * x1[-1]]
        # 对应用通道注意力后的 x1 两部分分别应用空间注意力机制并相乘
        x1 = [self.sa(x1[-2]) * x1[-2], self.sa(x1[-1]) * x1[-1]]
        # 将处理后的 x0 两部分在第 3 个维度上拼接
        x0 = torch.cat(x0, dim=3)
        # 将处理后的 x1 两部分在第 3 个维度上拼接
        x1 = torch.cat(x1, dim=3)
        # 将拼接后的 x0 和 x1 在第 2 个维度上拼接
        x3 = torch.cat((x0, x1), dim=2)

        # 将两部分处理结果相加
        x = x3 + x4
        return x

if __name__ == '__main__':
    input_tensor = torch.rand(1, 32, 50, 50)
    lpa = LPA(in_channel=32)
    output_tensor = lpa(input_tensor)
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {output_tensor.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")