import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://arxiv.org/abs/2409.13435
    论文题目：PlainUSR: Chasing Faster ConvNet for Eﬃcient Super-Resolution（ACCV 2024）
    中文题目：PlainUSR：追求更快的卷积神经网络以实现高效超分辨率
    讲解视频：
        局部重要性注意力（Local Importance-based Attention，LIA）：
             解决问题：一阶注意力机制性能弱：仅通过一次逐元素乘法进行计算。这种简单的操作方式导致其在捕捉图像特征、
                            区分有用和无用信息等方面能力有限，使得模型性能表现较弱，难以满足对图像超分辨率质量的要求。
                     二阶注意力机制计算复杂度高：像Self-Attention这类二阶空间注意力机制，虽然通过两次矩阵乘法能实现更
                            全面的信息交互和长距离建模，但这种计算方式会带来二次复杂度，进而增加模型的运行时间和计算资源消耗，
                            在对计算资源和延迟要求严格的轻量级超分辨率模型中难以适用。 
             实现方式：
"""
class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        # 调用父类构造函数初始化网络层
        super(SoftPooling2D, self).__init__()
        # 定义平均池化操作，不计入填充部分的计算
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        # 对输入张量应用指数运算
        x_exp = torch.exp(x)
        # 对指数处理后的张量进行平均池化
        x_exp_pool = self.avgpool(x_exp)
        # 对输入与指数结果相乘后进行平均池化
        x = self.avgpool(x_exp * x)
        # 返回最终的加权平均结果
        return x / x_exp_pool

class LocalAttention(nn.Module):
    def __init__(self, channels, f=16):
        # 初始化模块，设置通道数和中间层维度
        super().__init__()
        # 定义注意力机制的主要处理流程
        self.body = nn.Sequential(
            # 使用1x1卷积调整通道数
            nn.Conv2d(channels, f, 1),
            # 应用Soft Pooling来捕捉重要性信息
            SoftPooling2D(7, stride=3),
            # 通过3x3卷积进一步处理，步长为2
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            # 恢复到原始通道数
            nn.Conv2d(f, channels, 3, padding=1),
            # 使用Sigmoid激活函数生成权重
            nn.Sigmoid(),
        )

        """
            定义门控机制：
            为避免步长卷积和双线性插值带来的伪影，重新校准局部重要性，采用门控机制对局部重要性进行特征优化。
        """
        self.gate = nn.Sequential(
            nn.Sigmoid(),  # 使用Sigmoid作为门控激活函数
        )

    def forward(self, x):

        # 对输入的第一通道应用门控机制
        g = self.gate(x[:, :1].clone())

        # 将body处理的结果调整大小以匹配输入尺寸
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        # 返回输入与生成的权重和门控值的乘积
        return x * w * g

if __name__ == "__main__":
    # 创建一个形状为(1, 32, 64, 64)的随机输入张量
    input = torch.randn(1, 32, 64, 64)
    # 初始化局部注意力机制模块
    LA = LocalAttention(32)
    # 计算输出
    output = LA(input)
    # 打印输入和输出的尺寸
    print('input_size:', input.size())
    print('output_size:', output.size())
    # 提示信息
    print("布尔大学士 提醒您：代码无误~~~~")