import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://www.sciencedirect.com/science/article/pii/S1569843224001213
    论文题目：Robust change detection for remote sensing images based on temporospatial interactive attention module （2024 一区）
    中文题目：基于时空交互注意力模块的遥感图像稳健变化检测 （2024 一区）
    讲解视频：https://www.bilibili.com/video/BV1yvoWYxEDk/
    时空交互注意力模块（Temporospatial Interactive Attentive Modul,TIAM）：
        实际意义：①几何视角旋转问题：从不同角度观察同一区域时，会产生误导性干扰，导致语义错误解读。
                ②时间风格差异问题：不同时间拍摄图像时的光照、天气和季节变化,这些差异会导致图像产生伪变化
        实现方式：以代码为准。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class SpatiotemporalAttention(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False):
        super(SpatiotemporalAttention, self).__init__()
        assert dimension in [2, ]  # 维度校验[1](@ref)
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # 中间通道数初始化（类似VGG的通道缩减策略[1,2](@ref)）
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:  # 通道数下限保护
                self.inter_channels = 1

        # 特征变换模块组（包含BN和1x1卷积，类似AlexNet的通道调整[2](@ref)）
        self.g1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),  # 标准化层加速收敛[4](@ref)
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)  # 1x1卷积降维
        )
        self.g2 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )

        # 输出变换模块（包含卷积和BN，类似LeNet的全连接转换[2](@ref)）
        self.W1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),  # 升维恢复通道数
            nn.BatchNorm2d(self.in_channels)
        )
        self.W2 = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )

        # 注意力计算模块（theta和phi生成注意力特征）
        self.theta = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        # 特征变换（空间维度展开）
        g_x11 = self.g1(x1).reshape(batch_size, self.inter_channels, -1)  # [N, C', H*W]
        g_x21 = self.g2(x2).reshape(batch_size, self.inter_channels, -1)  # [N, C', H*W]

        # 图中Q1 和 Q2
        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)  # [N, C', H*W]
        theta_x2 = theta_x1.permute(0, 2, 1)  # 转置得到[N, H*W, C']
        # 图中K1 和 K2
        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)  # [N, C', H*W]
        phi_x2 = phi_x1.permute(0, 2, 1)  # 转置得到[N, H*W, C']

        # 时空注意力矩阵计算（包含矩阵相乘和维度置换）
        energy_time_1 = torch.matmul(theta_x1, phi_x2)  # [N, C', C']
        energy_time_2 = energy_time_1.permute(0, 2, 1)  # [N, C', C']
        energy_space_1 = torch.matmul(theta_x2, phi_x1)  # [N, H*W, H*W]
        energy_space_2 = energy_space_1.permute(0, 2, 1)  # [N, H*W, H*W]

        # 注意力权重归一化
        # 图中softmax
        energy_time_1s = F.softmax(energy_time_1, dim=-1)  # 时间维度归一化
        energy_time_2s = F.softmax(energy_time_2, dim=-1)
        energy_space_2s = F.softmax(energy_space_1, dim=-2)  # 空间维度归一化
        energy_space_1s = F.softmax(energy_space_2, dim=-2)

        # 注意力特征融合（矩阵相乘实现特征加权）
        # 图中最后的相乘
        y1 = torch.matmul(torch.matmul(energy_time_2s, g_x11), energy_space_2s).contiguous()  # [N, C', H*W]
        y2 = torch.matmul(torch.matmul(energy_time_1s, g_x21), energy_space_1s).contiguous()  # [N, C', H*W]

        # 恢复空间维度并残差连接（类似ResNet的跳跃连接[1,4](@ref)）
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])  # [N, C', H, W]
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        return x1 + self.W1(y1), x2 + self.W2(y2)  # 残差连接增强特征

if __name__ == '__main__':
    x1 = torch.randn(1, 64, 32, 32)
    x2 = torch.randn(1, 64, 32, 32)

    sp_full_not_shared = SpatiotemporalAttention(in_channels=64)
    output1, output2 = sp_full_not_shared(x1, x2)
    print("Input1 size:", x1.size())
    print("Input2 size:", x2.size())
    print("Output1 size:", output1.size())
    print("Output2 size:", output2.size())
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")