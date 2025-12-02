import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://ieeexplore.ieee.org/abstract/document/10379168/
    论文题目：Wavelet and Adaptive Coordinate Attention Guided Fine-Grained Residual Network for Image Denoising
    中文题目：用于图像去噪的小波与自适应坐标注意力引导细粒度残差网络
    讲解视频：https://www.bilibili.com/video/BV1dE1aBbE8k/
        自适应坐标注意力机制（Adaptive Coordinate Attention, ACA）
            实际意义：①细微噪声难以捕获问题：卷积依赖局部感受野和通道加权，难以准确识别局部性噪声分布，对低强度、细颗粒噪声的抑制效果有限。
                    ②空间位置信息丢失问题：常见通道注意力机制能在通道维度建模特征重要性，但忽略了空间位置信息（H、W方向的结构关系）。
                    ③噪声类型与强度差异适应性不足问题：不同图像的噪声特性差异较大（如均值、方差、分布形态不同），固定权重注意力机制缺乏动态调节能力。
            实现方式：通过“方向性全局池化 + 可学习缩放因子 + 融合归一化”的方式实现自适应空间定位，能在不同噪声场景下自动调整关注区域，
                        实现精确噪声捕获与边缘细节保持。
"""

class AdaptiveCoordAtt(nn.Module):
    def __init__(self, in_channels, reduction=16, alpha=0.9):
        super(AdaptiveCoordAtt, self).__init__()
        self.in_channels = in_channels           # 输入通道数
        self.reduction = reduction               # 通道压缩比例
        self.mid_channels = max(8, in_channels // reduction)  # 中间层通道数（瓶颈层）
        self.alpha = alpha                       # 缩放系数，用于调节注意力响应强度

        # 共享卷积层（相当于一个轻量 MLP）：1×1卷积 + BN + ReLU
        # 用于同时提取 H、W 方向的联合特征
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )

        # 分别为 H 和 W 方向恢复到原始通道数的卷积
        self.conv_h = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # 输入张量形状：[B, C, H, W]
        b, c, h, w = x.size()

        # 1️⃣ 沿 H 方向进行全局平均池化，得到形状 [B, C, H, 1]
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        # 2️⃣ 沿 W 方向进行全局平均池化，得到形状 [B, C, 1, W]
        x_w = F.adaptive_avg_pool2d(x, (1, w))
        # 调整维度顺序，使得 W 分支可以和 H 分支拼接（变成 [B, C, W, 1]）
        x_w = x_w.permute(0, 1, 3, 2)

        # 3️⃣ 沿空间维度拼接（H+W），形成联合特征 [B, C, (H+W), 1]
        y = torch.cat([x_h, x_w], dim=2)
        # 4️⃣ 通过共享 1×1 卷积提取融合特征
        y = self.shared_conv(y)

        # 5️⃣ 将融合特征再分为 H 分支和 W 分支
        x_h_out, x_w_out = torch.split(y, [h, w], dim=2)
        # 将 W 分支重新调整维度为 [B, C, 1, W]
        x_w_out = x_w_out.permute(0, 1, 3, 2)

        # 6️⃣ 通过各自卷积映射回输入通道维，并使用 Sigmoid 得到注意力权重
        # α 控制注意力强度（小于1时降低响应）
        a_h = self.conv_h(x_h_out * self.alpha).sigmoid()  # 形状 [B, C, H, 1]
        a_w = self.conv_w(x_w_out * self.alpha).sigmoid()  # 形状 [B, C, 1, W]

        # 7️⃣ 组合 H 和 W 的注意力权重，并逐通道加权输入特征
        out = x * (a_h + a_w)
        # 输出张量形状与输入相同 [B, C, H, W]
        return out

if __name__ == "__main__":
    # 构造输入张量 [batch=2, 通道数=32, 高=宽=50]
    x = torch.randn(2, 32, 50, 50)
    # 实例化模块，通道数32，通道压缩比2
    model = AdaptiveCoordAtt(in_channels=32, reduction=2)
    y = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {y.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")