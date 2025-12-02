import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/abs/2412.16986
    论文题目：Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection（AAAI 2025）
    中文题目：基于风车状卷积和尺度动态损失的红外小目标检测(AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV1X9FBenEcs/
        风车形状卷积模块（Pinwheel-Shaped Convolutional Module，PSCM）：
            理论支撑：PConv模块通过独特设计对齐弱小目标像素高斯分布。
            实现方式：非对称填充，针对目标不同区域设置水平和垂直卷积核，从不同方向捕捉目标像素，重点关注中心关键区域。
                    分组卷积扩大感受野，能覆盖目标及周围相关区域，整合分散像素信息。
                    通过多方向并行卷积结果拼接，全面捕捉不同角度特征，融合后更完整地匹配目标像素的高斯分布特性，提升特征提取能力 。 
"""

def autopad(k, p=None, d=1):
    """根据卷积核大小、膨胀率计算填充大小，以保持输出形状与输入相同。"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # 计算膨胀卷积的有效卷积核大小
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动计算填充大小
    return p

class Conv(nn.Module):
    """标准卷积层，包含卷积、批归一化和激活函数。"""
    default_act = nn.SiLU()  # 默认激活函数为SiLU
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """初始化卷积层，参数包括输入通道数、输出通道数、卷积核大小、步幅、填充、分组、膨胀率和激活函数。"""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)  # 卷积层
        self.bn = nn.BatchNorm2d(c2)  # 批归一化层
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()  # 激活函数

    def forward(self, x):
        """前向传播，对输入张量应用卷积、批归一化和激活函数。"""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """融合前向传播，只应用卷积和激活函数，用于模型推理优化。"""
        return self.act(self.conv(x))


class PSConv(nn.Module):
    """风车形卷积层，使用非对称填充方法。"""
    def __init__(self, c1, c2, k=3, s=1):
        """初始化风车形卷积层，参数包括输入通道数、输出通道数、卷积核大小和步幅。"""
        super().__init__()

        # 零填充：分别指定左、右、上、下四条边的填充量
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]  # 四个方向的非对称填充
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]  # 创建四个填充层

        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)  # 水平方向卷积
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)  # 垂直方向卷积

        self.cat = Conv(c2, c2, 2, s=1, p=0)  # 合并卷积结果

    def forward(self, x):
        """前向传播，对输入张量应用风车形卷积。"""
        yw0 = self.cw(self.pad[0](x))  # 水平方向，第一种填充
        yw1 = self.cw(self.pad[1](x))  # 水平方向，第二种填充

        yh0 = self.ch(self.pad[2](x))  # 垂直方向，第一种填充
        yh1 = self.ch(self.pad[3](x))  # 垂直方向，第二种填充

        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))  # 拼接四个方向的卷积结果，并通过合并卷积层


if __name__ == "__main__":
    module = PSConv(c1=64, c2=128)  # 创建风车形卷积层实例
    input_tensor = torch.randn(1, 64, 128, 128)  # 创建随机输入张量
    output_tensor = module(input_tensor)  # 执行前向传播
    print('Input size:', input_tensor.size())  # 打印输入张量形状
    print('Output size:', output_tensor.size())  # 打印输出张量形状