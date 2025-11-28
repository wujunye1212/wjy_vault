import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://arxiv.org/abs/2507.01838
    论文题目：MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices (ICCV 2025)
    中文题目：MobileIE：一种用于移动设备实时图像增强的超轻量级高效卷积神经网络 (ICCV 2025)
    讲解视频：https://www.bilibili.com/video/BV1NMJDzUEAH/
    分层双路径注意力机制（Hierarchical Dual-Path Attention，HDPA）：
        实际意义：①全局与局部特征兼顾问题：在边缘设备上，轻量化图像增强模型无法兼顾全局上下文信息（保证结构一致性）和局部细节特征（增强边缘和纹理）。
                ②轻量化与硬件友好性：Transformer 和扩散模型虽然增强效果强，但自注意力计算复杂度高，不适合移动端。
                ③特征精度与稳定性提升：由于轻量化模型参数量有限，特征提取容易不充分，导致增强效果欠佳。
        实现方式：全局路径 → 局部路径 → 特征融合
"""

class MBRConv1X1(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv1X1, self).__init__()
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.rep_scale = rep_scale  # 通道扩展倍率

        # 1×1 卷积，将输入通道映射到 (输出通道 × 扩展倍率)
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, kernel_size=1)

        # 对扩展后的通道进行批归一化
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )

        # 输出卷积：输入通道是 (2 × 输出通道 × 扩展倍率)，输出恢复到 out_channels
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, kernel_size=1)
        self.conv_out.weight.requires_grad = False  # 固定该卷积层的参数，不进行训练

        # 新增的可学习参数，用来增强 conv_out 的卷积权重
        self.weight1 = nn.Parameter(torch.zeros_like(self.conv_out.weight))
        nn.init.xavier_normal_(self.weight1)  # 用 Xavier 初始化 weight1

    def forward(self, inp):
        # 第一步：输入经过 1×1 卷积，通道扩展
        x1 = self.conv(inp)  # 形状 [B, out*rep, H, W]

        # 第二步：拼接卷积结果与其 BN 归一化结果
        x = torch.cat([x1, self.conv_bn(x1)], dim=1)  # 形状 [B, 2*out*rep, H, W]

        # 第三步：动态更新卷积核权重 (固定权重 + 可学习权重)
        final_weight = self.conv_out.weight + self.weight1

        # 第四步：手动执行卷积运算
        out = F.conv2d(x, final_weight, bias=self.conv_out.bias, stride=1, padding=0, dilation=1, groups=1)
        return out  # 输出形状 [B, out, H, W]


class HDPA(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(HDPA, self).__init__()
        self.channels = channels  # 输入和输出的通道数

        # 全局注意力分支：先经过 MBRConv1X1，再做全局平均池化和 Sigmoid
        self.globalatt = nn.Sequential(
            MBRConv1X1(channels, channels, rep_scale=rep_scale),  # 输出 [B, C, H, W]
            nn.AdaptiveAvgPool2d(1),  # 压缩为全局特征 [B, C, 1, 1]
            nn.Sigmoid()  # 映射到 (0,1)，得到注意力权重
        )

        # 局部注意力分支：输入单通道特征，输出 C 通道的局部注意力权重
        self.localatt = nn.Sequential(
            MBRConv1X1(1, channels, rep_scale=rep_scale),  # 输出 [B, C, H, W]
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算全局注意力权重
        x1 = self.globalatt(x)  # [B, C, 1, 1]

        # 将全局注意力作用到原特征，并在通道维度取最大值
        max_out, _ = torch.max(x1 * x, dim=1, keepdim=True)  # [B, 1, H, W]

        # 计算局部注意力权重
        x2 = self.localatt(max_out)  # [B, C, H, W]

        # 最终融合：全局注意力 × 局部注意力 × 原特征
        x3 = torch.mul(x1, x2) * x  # 广播相乘，逐点增强有效特征
        return x3

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)  # 构造输入张量 [B=1, C=32, H=50, W=50]
    model = HDPA(channels=32)  # 初始化 HDPA 模块
    output = model(x)  # 前向传播
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")