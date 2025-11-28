import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2405.05497
    论文题目：Multi-Level Feature Fusion Network for Lightweight Stereo Image Super-Resolution (CVPR 2024)
    中文题目：轻量级立体图像超分辨率的多级特征融合网络（CVPR 2024）
    讲解视频：https://www.bilibili.com/video/BV1UL6WYHEnD/
        交叉视图特征交互（Cross-View Feature Interaction,  CVIM）
             提出问题：冗余跨视图特征交互对超分辨率性能提升贡献不大，但会显著增加计算复杂度。
             理论研究：实现左右视图间跨视图特征交互和融合功能，通过输入特征进行深度卷积、逐点卷积、缩放点积注意力机制，
                     最终将跨视图融合特征和原始视图内特征按照可训练的比例进行融合，得到视图特征，提供丰富有效特征表示，
                     提高超分辨率性能，同时通过改进减少计算复杂度。
"""

class CVIM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5  # 缩放因子，用于归一化

        # 定义左侧第一组卷积层
        self.l_proj1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),  # 1x1卷积
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)  # 3x3深度卷积
        )
        # 定义右侧第一组卷积层
        self.r_proj1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),  # 1x1卷积
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)  # 3x3深度卷积
        )

        # 定义左侧第二组卷积层
        self.l_proj2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),  # 1x1卷积
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)  # 3x3深度卷积
        )
        # 定义右侧第二组卷积层
        self.r_proj2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),           # 1x1卷积
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)  # 3x3深度卷积
        )

        # 定义左侧第三个卷积层
        self.l_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)  # 1x1卷积
        # 定义右侧第三个卷积层
        self.r_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)  # 1x1卷积

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(x_l).permute(0, 2, 3, 1)    # B, H, W, c，左侧特征转置
        Q_r_T = self.r_proj1(x_r).permute(0, 2, 1, 3)  # B, H, c, W，右侧特征转置
        # 计算注意力矩阵 (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c，左侧特征转置
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c，右侧特征转置
        # 计算从右到左的特征映射
        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        # 计算从左到右的特征映射
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c

        # 缩放并转换维度
        F_r2l = self.l_proj3(F_r2l.permute(0, 3, 1, 2))
        F_l2r = self.r_proj3(F_l2r.permute(0, 3, 1, 2))
        return x_l + F_r2l + x_r + F_l2r  # 返回输出


if __name__ == '__main__':
    input1 = torch.randn(1, 32, 64, 64)  # 生成随机输入张量1
    input2 = torch.randn(1, 32, 64, 64)  # 生成随机输入张量2
    # 初始化CVIM模块并设定通道维度
    CVIM_module = CVIM(32)
    output = CVIM_module(input1, input2)  # 计算输出

    print("Input size:", input1.size(),input2.shape)
    print("Output size:", output.size())
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")