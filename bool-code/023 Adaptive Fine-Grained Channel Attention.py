import math
import torch
from torch import nn

"""
论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0893608024002387
论文题目：Unsupervised Bidirectional Contrastive Reconstruction and Adaptive Fine-Grained Channel Attention Networks for image dehazing (2024 一区Top)
中文题目：用于图像去雾的无监督双向对比重建和自适应细粒度通道注意网络
讲解视频：https://www.bilibili.com/video/BV1rP2NYBELd/

自适应细粒度通道注意力（Adaptive Fine-Grained Channel Attention, FCA）机制作用：
1. 全局与局部信息的有效结合：FCA机制通过捕捉不同尺度上的全局和局部信息之间的相互依赖关系，并促进二者之间的交互。这种交互有助于更精确地描述不同层次的信息关联性。
2. 特征权重的自适应分配：利用可学习参数动态融合全局和局部信息，FCA能够实现通道权重的自适应分配。这意味着它可以根据具体情况调整每个通道的重要性，从而增强去雾效果。
3. 解决SE通道注意力机制的局限性：传统的Squeeze-and-Excitation (SE) 通道注意力机制主要关注于全局信息而忽略了局部信息的作用。相比之下，FCA机制不仅考虑到了全局信息，还有效地整合了局部信息，
                            解决了SE机制因缺乏对局部信息的利用而导致的特征权重分配不准确的问题。
综上所述，自适应细粒度通道注意力机制通过改善通道间的信息流动和权重分配策略，增强了图像去雾网络的表现，尤其是在处理复杂的雾霾条件下，能够产生更加清晰、自然且细节保留更好的去雾图像。
"""

# 定义Mix类，继承自nn.Module
class Mix(nn.Module):
    # 初始化方法，设置初始混合权重m
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()  # 调用父类的初始化方法
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)  # 创建可学习参数w

        self.w = w  # 将w作为实例变量保存
        self.mix_block = nn.Sigmoid()  # 定义Sigmoid激活函数层

    # 前向传播方法，输入两个特征图fea1和fea2
    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)  # 计算混合因子
        # 根据混合因子融合fea1和fea2
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out  # 返回融合后的特征图

class FCAttention(nn.Module):
    # 初始化方法，设置通道数channel，以及b和gamma参数
    def __init__(self, channel, b=1, gamma=2):
        super(FCAttention, self).__init__()  # 调用父类的初始化方法
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化层

        # 计算一维卷积核大小k
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        # 定义一维卷积层
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)

        # 定义二维卷积层（全连接层）
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()  # 定义Sigmoid激活函数层
        self.mix = Mix()  # 实例化Mix对象

    # 前向传播方法，输入为input
    def forward(self, input):
        U = self.avg_pool(input)  # 对输入进行全局平均池化

        # 对U进行挤压和转置后进行一维卷积，然后再次转置回来
        Ugc = self.conv1(U.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)  # (batch_size, channels, 1)
        # 对U进行二维卷积，然后挤压和转置
        Ulc = self.fc(U).squeeze(-1).transpose(-1, -2)  # (batch_size, 1, channels)

        # 计算矩阵乘法并求和，然后通过Sigmoid激活函数
        out1 = torch.sum(torch.matmul(Ugc, Ulc), dim=1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, channels, 1, 1)
        out1 = self.sigmoid(out1)

        # 类似地计算另一个输出out2
        out2 = torch.sum(torch.matmul(Ulc.transpose(-1, -2), Ugc.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)

        # 使用Mix层融合out1和out2
        out = self.mix(out1, out2)

        # 再次进行一维卷积、转置和扩展维度，并通过Sigmoid激活函数
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        W = self.sigmoid(out)

        # 返回与W相乘的结果
        return input * W

# 主程序入口
if __name__ == '__main__':
    model = FCAttention(channel=64)  # 创建FCAttention模型实例，通道数为64

    input = torch.rand(1, 64, 128, 128)  # 创建随机输入张量

    output = model(input)  # 将输入传递给模型，得到输出
    print('input_size:', input.size())  # 打印输入尺寸
    print('output_size:', output.size())  # 打印输出尺寸

    print("抖音、B站、小红书、CSDN同号")  # 输出社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 提示信息