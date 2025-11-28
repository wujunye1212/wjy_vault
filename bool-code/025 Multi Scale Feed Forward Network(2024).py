import torch
import torch.nn as nn
import torch.nn.functional as F

"""
论文地址：https://ieeexplore.ieee.org/abstract/document/10445289/
论文题目：Hybrid Convolutional and Attention Network for Hyperspectral Image Denoising (2024)
中文题目：用于高光谱图像去噪的混合卷积和注意力网络
讲解视频：https://www.bilibili.com/video/BV1Zv26YPE2A/
        多尺度前馈网络模块（Multi-Scale Feed-Forward Network）
        作用：对CAFM的输出特征进行处理，可以聚合多尺度特征并增强非线性特征，专注于丰富上下文信息。
        原理：1、提取的特征元素乘法Gate机制以增强非线性变换。
             2、多尺度膨空洞卷积用于多尺度特征提取。
             【替换Transformer】
"""
class Multi_Scale_Feed_Forward_Network(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(Multi_Scale_Feed_Forward_Network, self).__init__()

        # 计算隐藏层特征的数量，基于输入维度和扩张因子
        hidden_features = int(dim * ffn_expansion_factor)

        # 定义一个3D卷积层用于将输入维度映射到隐藏层维度（扩展后的）5维
        self.project_in = nn.Conv3d(dim, hidden_features * 3, kernel_size=(1, 1, 1), bias=bias)

        # 第一个深度可分离卷积层，使用3x3x3的核大小，并且步长为1，空洞率为1，padding为1，确保输出尺寸不变
        self.dwconv1 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3, 3, 3), stride=1, dilation=1,
                                 padding=1, groups=hidden_features, bias=bias)

        # 第二个深度可分离卷积层，但这是2D的，因此它处理的是从3D转换来的2D数据。使用3x3的核大小，步长为1，空洞率为2，padding为2。
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3), stride=1, dilation=2, padding=2,
                                 groups=hidden_features, bias=bias)

        # 第三个深度可分离卷积层，同样为2D，设置与第二个类似但空洞率不同。
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3), stride=1, dilation=3, padding=3,
                                 groups=hidden_features, bias=bias)

        # 最后一个3D卷积层，用于将隐藏层维度重新映射回原始输入维度
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=(1, 1, 1), bias=bias)

    # 前向传播函数，定义了数据如何通过网络
    def forward(self, x):
        # input = torch.randn(1, 32, 8, 8)
        # 在第二维度上添加一个维度，使得可以进行3D卷积操作 4  5
        x = x.unsqueeze(2)  # torch.Size([1, 32, 1, 8, 8])

        # 使用project_in层将输入映射到隐藏层空间
        x = self.project_in(x)  # torch.Size([1, 192, 1, 8, 8])
        # 将经过第一个卷积层后的张量分割成三部分，每部分对应不同的后续处理
        x1, x2, x3 = x.chunk(3, dim=1)  # torch.Size([1, 64, 1, 8, 8])

        # 对第一部分执行深度可分离3D卷积，并移除额外增加的维度
        x1 = self.dwconv1(x1).squeeze(2)

        # 对第二、第三部分,首先需要移除之前增加的维度，再执行2D深度可分离卷积。【空洞率不同】
        x2 = self.dwconv2(x2.squeeze(2))
        x3 = self.dwconv3(x3.squeeze(2))

        # 使用GELU激活函数对x1进行激活，然后乘以x2和x3的结果
        x = F.gelu(x1) * x2 * x3

        # 为最后的输出准备，再次在第二维度上添加一个维度
        x = x.unsqueeze(2)
        # 使用project_out层将结果映射回原始输入维度
        x = self.project_out(x)
        # 移除之前为了适应3D卷积而添加的维度
        x = x.squeeze(2)
        return x


# 主程序入口
if __name__ == '__main__':
    # 创建随机输入张量
    input = torch.randn(1, 32, 8, 8)

    # 实例化多尺度前馈网络模型，指定维度为32，扩张因子为2，使用偏置
    model = Multi_Scale_Feed_Forward_Network(dim=32, ffn_expansion_factor=2, bias=True)

    # 模型前向传播，获取输出
    output = model(input)

    # 打印输入和输出张量的形状
    print('input_size:', input.size())
    print('output_size:', output.size())

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")