import torch
import torch.nn as nn
from torch import Tensor

"""
    论文地址：https://www.sciencedirect.com/science/article/pii/S0957417425022559
    论文题目：Channel grouping vision transformer for lightweight fruit and vegetable recognition（2025 一区TOP）
    中文题目：用于轻量化果蔬识别的通道分组视觉 Transformer（CGViT）（2025 一区TOP）
    讲解视频：https://www.bilibili.com/video/BV1tqexzXE74/
        通道分组折半卷积模块（Channel Grouping Half-convolution，CGHF）：
            实际意义：①传统 CNN 难以针对性提取多层级特征：果蔬图像存在三类关键特征（表面属性如颜色、形状、纹理、深度），无法“差异化”捕捉不同层级特征，导致相似类别难以区分。
                    ②全通道卷积导致计算与参数冗余：传统卷积对特征图所有通道进行相同操作，产生大量冗余计算，特征部分通道信息重复，无需全通道密集计算。
            实现方式：①将特征图通道划分为3组，每组独立卷积实现对判别性特征的敏感度，保证不同特征（颜色、纹理、深层语义）的多样性和完整性。
                    ②仅对一半通道进行卷积，另一半保持不变，减少 FLOPs和内存访问量。
"""

class HalfConv(nn.Module):
    def __init__(self, dim, n_div=2):
        super().__init__()
        # 将通道数划分为两部分：一部分做卷积，一部分保持不变
        self.dim_conv3 = dim // n_div   # 需要进行卷积的通道数
        self.dim_untouched = dim - self.dim_conv3  # 保持不变的通道数
        # 定义一个 3x3 的卷积操作，仅作用在 self.dim_conv3 个通道上
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # 将输入特征图在通道维度上拆分成两部分：一部分做卷积，一部分保持原样
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        # 对 x1 进行 3x3 卷积
        x1 = self.partial_conv3(x1)
        # 将卷积后的 x1 与保持不变的 x2 拼接回去
        x = torch.cat((x1, x2), 1)
        return x

class CGHalfConv(nn.Module):
    def __init__(self, dim):
        super(CGHalfConv, self).__init__()
        # 将输入通道数尽量平均分成 3 份
        self.div_dim = int(dim / 3)     # 每份的通道数
        self.remainder_dim = dim % 3    # 余下的通道数
        # 定义三个 HalfConv 模块，分别作用于三个子通道段
        """
            主页的注意力机制在这里随便加！
        """
        self.p1 = HalfConv(self.div_dim, 2)
        self.p2 = HalfConv(self.div_dim, 2)
        self.p3 = HalfConv(self.div_dim + self.remainder_dim, 2)

    def forward(self, x):
        # 保留输入用于残差连接
        y = x
        # 将输入在通道维度上拆分为三部分
        x1, x2, x3 = torch.split(x, [self.div_dim, self.div_dim, self.div_dim + self.remainder_dim], dim=1)
        # 分别送入对应的 HalfConv 模块
        x1 = self.p1(x1)
        x2 = self.p2(x2)
        x3 = self.p3(x3)
        # 拼接处理后的三个部分
        x = torch.cat((x1, x2, x3), 1)
        # 加上残差，增强训练稳定性和特征表达能力
        return x + y

if __name__ == "__main__":
    # 构造一个随机输入张量：batch=1, 通道数=32, 高=50, 宽=50
    x = torch.randn(1, 32, 50, 50)
    model = CGHalfConv(dim=32)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")