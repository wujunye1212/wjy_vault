import torch
import torch.nn as nn
from timm.models.layers import DropPath

"""
论文地址：https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Rewrite_the_Stars_CVPR_2024_paper.pdf
论文题目：Rewrite the Stars（CVPR 2024）
讲解视频：https://www.bilibili.com/video/BV16xwGehErG/
  深度学习：通常先将低维特征线性投影到高维空间，再使用激活函数（例如ReLU、GELU等）引入非线性。   
  机器学习：通过核技巧同时获得高维度和非线性。例如，高斯核函数通过泰勒展开产生无限维特征空间。
  在这项工作中，证明star操作可以在低维输入中获取高维和非线性的特征空间，轻量级的多阶通道重分配模块，可以自适应地重分配高维隐藏空间通道特征，
                                                        这个过程允许模型自适应地调整每个通道的重要性，从而优化整个网络信息流动。

"""
class ConvBN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        if p is None:
            # 如果卷积核大小为整数，则使用卷积核的一半作为自动填充值；
            # 若卷积核大小为列表，则对每个卷积核尺寸计算其一半作为对应的填充值。
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)  # 初始化二维卷积层
        self.bn = nn.BatchNorm2d(c2)  # 初始化批归一化层
        self.act = nn.SiLU()
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.bn(self.conv(x))              # V1 对输入张量应用卷积、批归一化
        # return self.act(self.bn(self.conv(x)))  # V2 对输入张量应用卷积、批归一化以及激活函数

class Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, g=dim)                   # 深度可分离卷积层

        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)          # 在希望保留输入数据的空间结构时，可以用1x1卷积来代替全连接层。
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)          # 在希望保留输入数据的空间结构时，可以用1x1卷积来代替全连接层。

        self.g = ConvBN(mlp_ratio * dim, dim, 1)  # Star
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)  # 另一个深度可分离卷积层

        self.act = nn.ReLU6()                                               # 使用ReLU6作为激活函数

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # 随机丢弃路径以防止过拟合

    def forward(self, x):
        input = x  # 保存原始输入

        x = self.dwconv(x)  # 应用第一个深度可分离卷积
        x1, x2 = self.f1(x), self.f2(x)  # 分别通过两个全连接层

        x = self.act(x1) * x2  # 将经过激活的第一个输出与第二个输出相乘

        x = self.g(x)  # star Conv
        x = self.dwconv2(x)  # 轻量级的多阶通道重分配模块，可以自适应地重分配高维隐藏空间通道特征，允许模型自适应地调整每个通道的重要性，从而优化整个网络信息流动。

        x = input + self.drop_path(x)  # 添加原始输入，并可能随机丢弃部分路径
        return x


if __name__ == '__main__':

    block = Star_Block(dim=32)  # 创建Star_Block实例，特征维度为32

    input = torch.rand(1, 32, 64, 64)  # 生成随机输入数据

    output = block(input)  # 将数据传入block处理
    print("input.shape:", input.shape)  # 打印输入形状
    print("output.shape:", output.shape)  # 打印输出形状

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")