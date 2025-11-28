import torch.nn as nn
import torch.nn.functional as F
import torch

"""
论文地址：https://ieeexplore.ieee.org/abstract/document/10445289/
论文题目：Restoring Images in Adverse Weather Conditions via Histogram Transformer (ECCV 2024)
中文题目：Histoformer：基于直方图 Transformer的恶劣天气条件图像恢复
讲解视频：https://www.bilibili.com/video/BV1mhmFYBE7U/
        双尺度前馈门控网络（Dual-scale Gated Feed-Forward）：
        在标准的FFN网络中使用单范围或单尺度卷积来增强局部上下文。然而往往忽略动态分布的天气引起的退化之间的相关性。
        因此，提出通过扩大核大小与利用膨胀机制可以提取多尺度信息。
"""
class Dual_scale_Gated_FFN(nn.Module):
    # 初始化函数，设置输入维度dim和FFN扩展因子ffn_expansion_factor，默认为2.0，并决定是否使用偏置bias
    def __init__(self, dim, ffn_expansion_factor=2.0, bias=True):
        super(Dual_scale_Gated_FFN, self).__init__()  # 调用父类nn.Module的初始化方法

        hidden_features = int(dim * ffn_expansion_factor)  # 计算隐藏层特征数

        # 输入投影层：使用1x1卷积核将输入维度变换到两倍隐藏层特征数
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # PixelShuffle层：用于上采样，参数2表示上采样的比例
        self.p_shuffle = nn.PixelShuffle(2)

        # 深度可分离卷积层5x5：处理部分通道的数据，保持通道间独立性
        self.dwconv_5 = nn.Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=5, stride=1, padding=2, groups=hidden_features // 4, bias=bias)

        # 带有膨胀率的深度可分离卷积层3x3：同样处理部分通道数据，但通过设置dilation=2来扩大感受野
        self.dwconv_dilated2_1 = nn.Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=3, stride=1, padding=2, groups=hidden_features // 4, bias=bias, dilation=2)

        # PixelUnshuffle层：与PixelShuffle相对，用于下采样
        self.p_unshuffle = nn.PixelUnshuffle(2)

        # 输出投影层：再次使用1x1卷积核，将经过处理后的特征映射回原始输入维度
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    # 定义模型的前向传播过程
    def forward(self, x):
        x = self.project_in(x)  # 将输入通过输入投影层    torch.Size([1, 256, 32, 32])
        x = self.p_shuffle(x)  # 使用PixelShuffle进行上采样    # torch.Size([1, 64, 64, 64])

        x1, x2 = x.chunk(2, dim=1)  # 分割x为两份，每份包含一半的通道数

        x1 = self.dwconv_5(x1)  # 第一部分数据通过深度可分离卷积层5x5
        x2 = self.dwconv_dilated2_1(x2)  # 第二部分数据通过带有膨胀率的深度可分离卷积层3x3

        x = F.mish(x2) * x1  # 对第二部分应用Mish激活函数后与第一部分相乘   torch.Size([1, 32, 64, 64])

        x = self.p_unshuffle(x)  # 使用PixelUnshuffle进行下采样        torch.Size([1, 128, 32, 32])
        x = self.project_out(x)  # 最终输出通过输出投影层

        return x  # 返回最终输出

if __name__ == "__main__":
    # 创建Dual_scale_Gated_FFN模块的一个实例，指定输入维度为64
    model = Dual_scale_Gated_FFN(64)
    input = torch.randn(1, 64, 32, 32)  # 生成随机输入张量，形状为(1, 64, 32, 32)

    output = model(input)

    print('Input size:', input.size())  # 打印输入张量大小
    print('Output size:', output.size())  # 打印输出张量大小