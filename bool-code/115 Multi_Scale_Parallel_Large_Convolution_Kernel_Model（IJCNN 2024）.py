import torch
import torch.nn as nn
"""
    论文地址：https://arxiv.org/pdf/2305.17654
    论文题目：MixDehazeNet : Mix Structure Block For Image Dehazing Network（IJCNN 2024）
    中文题目：MixDehazeNet：用于图像去雾网络的混合结构块（IJCNN 2024）
    讲解视频：https://www.bilibili.com/video/BV1qNwXeMEGN/
        多尺度并行大核卷积模块（Multi-Scale Parallel Large Convolution Kernel，MSPLCK）：
            实际意义：基于CNN和Transformer的方法虽能利用大感受野提升性能，但去雾时会忽略图像多尺度特性。
                    基于Transformer的模型参数多、训练难，因此研究聚焦于基于CNN的方法，利用大卷积核获取大感受野和长距离建模能力。
            实现方式：通过批归一化处理原始特征图，利用不同大小的并行空洞卷积提取多尺度特征，大、中卷积关注大雾霾区域，小卷积恢复纹理细节，最后将融合特征与输入特征相加。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
            相似思路参考：https://www.bilibili.com/video/BV1ohK5e6Eb4/
"""
# 定义一个多尺度并行大卷积核模型的类，继承自nn.Module
class Multi_Scale_Parallel_Large_Convolution_Kernel_Model(nn.Module):
    def __init__(self, dim):
        super().__init__()  # 调用父类的初始化方法

        # 定义一个2D批归一化层
        self.norm1 = nn.BatchNorm2d(dim)

        # 定义一系列卷积层
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)  # 1x1卷积
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')  # 5x5卷积，反射填充

        # 深度卷积，使用不同的卷积核大小和膨胀率
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_5 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_3 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        # 多层感知机（MLP）模块，包含两个卷积层和一个GELU激活函数
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),  # 卷积层，改变通道数
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim * 4, dim, 1)  # 卷积层，恢复通道数
        )

    # 定义前向传播过程
    def forward(self, x):
        identity = x  # 保存输入用于残差连接

        x = self.norm1(x)  # 批归一化
        x = self.conv1(x)  # 1x1卷积
        x = self.conv2(x)  # 5x5卷积

        # 将三个不同卷积核的输出在通道维度上拼接
        x = torch.cat([self.conv3_7(x), self.conv3_5(x), self.conv3_3(x)], dim=1)

        x = self.mlp(x)  # 通过MLP模块
        x = identity + x  # 残差连接
        return x  # 返回输出

if __name__ == '__main__':
    # 实例化模型对象，指定输入通道数
    model = Multi_Scale_Parallel_Large_Convolution_Kernel_Model(dim=32)
    input = torch.randn(8, 32, 64, 64)
    # 执行前向传播，获取输出
    output = model(input)
    # 打印输入和输出的张量形状
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')

    print('input_size:', input.size())
    print('output_size:', output.size())
    print("公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")

