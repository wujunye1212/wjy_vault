from torch import nn
import torch
import torch.nn.functional
import torch.nn.functional as F
'''
论文题目：用于医学图像分割的通道先验卷积注意力
通道先验卷积注意力即插即用模块：CPCA        Arxiv 2023
本文主要内容：
低对比度和显着器官形状变化等特征经常出现在医学图像中。
医学图像中分割性能的提高受到现有注意力机制普遍适应能力不足的限制。
本文提出了一种高效的通道先验卷积注意力（CPCA）方法，
支持注意力权重在通道和空间维度上的动态分布。
通过采用多尺度深度卷积模块，在保留通道先验的同时有效地提取空间关系。
CPCA 拥有专注于全局和重要区域特征信息的能力。

简单介绍CPCA模块： 
通道先验卷积注意力 （CPCA） 具有包括通道注意力和空间注意力的顺序放置的整体结构。
特征图的空间信息由通道注意力通过 average pooling、max pooling 等操作进行聚合。
空间信息随后通过共享的 MLP（多层感知机）进行处理并添加以生成通道注意力图。
通道先验是通过输入特征和通道注意力图的元素相乘获得的。
随后，将通道先验输入到深度卷积模块中，生成空间注意力图。
卷积模块接收空间注意力图以进行通道混合。
最终，通过将通道混合结果与通道先验的元素相乘，获得优化后的特征作为输出。
通道混合过程有助于增强特征的表示。

适用于：医学图像分割等所有CV2维任务通用的注意力模块
'''
class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x

class CPCA(nn.Module):

    def __init__(self, in_channels, out_channels, channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out

# 输入 B C H W,  输出B C H W
if __name__ == "__main__":
    # 创建 CPCA 模块的实例
    cpca_module = CPCA(in_channels=64, out_channels=64)
    input_tensor = torch.randn(1, 64, 128, 128)
    # 执行前向传播
    output_tensor = cpca_module(input_tensor)
    print('Input size:', input_tensor.size())
    print('Output size:', output_tensor.size())