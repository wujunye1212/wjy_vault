import torch
import torch.nn as nn
import torch.nn.functional as F

"""
论文地址：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf
论文题目：SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution（ECCV 2024）
讲解视频：https://www.bilibili.com/video/BV1sxcde4EpX/

  高效近似自注意力 (Efficient Approximation of Self-attention,EASA)
     为了获得低频信息，首先经过缩放因子为8的自适应最大池化操作D，接着传入一个3×3深度可分离卷积层来生成非局部结构信息，
     接着引入了 X 的方差作为空间信息的统计差异，并通过1×1卷积将其与非局部 Xs 合并得到调制特征，
     最后，聚合调制特征与输入特征X得到代表性结构信息 Xl。
     
"""

class EASA(nn.Module):
    def __init__(self, dim=36):
        super(EASA, self).__init__()
        # 定义1x1卷积层，将输入通道数从dim扩展到2*dim
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        # 定义1x1卷积层，保持通道数不变
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        # 定义1x1卷积层，保持通道数不变
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        # 定义深度可分离卷积层（Depth-wise Convolution），保持通道数不变
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        # 定义GELU激活函数
        self.gelu = nn.GELU()
        # 设置下采样因子
        self.down_scale = 8

        # 定义可学习参数alpha，初始化为全1
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        # 定义可学习参数belt，初始化为全0
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, X):
        _, _, h, w = X.shape  # 获取输入特征图的高度和宽度

        # 对X进行自适应最大池化操作，然后通过深度可分离卷积层
        x_s = self.dw_conv(F.adaptive_max_pool2d(X, (h // self.down_scale, w // self.down_scale)))

        # 计算X的方差 ，作为空间信息的统计差异
        x_v = torch.var(X, dim=(-2, -1), keepdim=True)

        # 计算局部细节估计
        Temp = x_s * self.alpha + x_v * self.belt

        x_l = X * F.interpolate(self.gelu(self.linear_1(Temp)), size=(h, w), mode='nearest')

        # 通过线性层2输出最终结果
        return self.linear_2(x_l)

# 输入 N C H W, 输出 N C H W
if __name__ == '__main__':
    # 创建随机输入张量，形状为(1, 32, 256, 256)
    input = torch.rand(1, 32, 256, 256)
    # 实例化EASA模型，设置通道数为32
    model = EASA(dim=32)
    # 前向传播，获取输出
    output = model(input)
    # 打印输入和输出的尺寸
    print('input_size:', input.size())
    print('output_size:', output.size())

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")