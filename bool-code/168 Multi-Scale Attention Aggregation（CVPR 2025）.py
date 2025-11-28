import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2405.10530
    论文题目：Efficient Visual State Space Model for Image Deblurring （CVPR 2025）
    中文题目：图像去模糊的高效视觉状态空间模型
    讲解视频：https://www.bilibili.com/video/BV1b1GEzmEZz/
        多尺度注意力聚合模块（Multi-Scale Attention Aggregation，MSAA）：
            实际意义：①多尺度特征融合不足：不同尺度的目标（如建筑、道路、植被等），传统Unet的跳跃连接仅简单拼接特征，未充分挖掘多尺度上下文信息。
                    ②空间与通道维度的特征表示：传统网络难以同时兼顾空间位置精度和通道语义重要性，导致分割边缘模糊。
            实现方式：①特征拼接：合并相邻层特征，形成多尺度特征。
                    ②空间路径：1×1卷积降维；3/5/7卷积捕获不同大小目标；池化+7×7卷积强化空间定位。
                    ③通道路径：全局平均池化压缩特征；1×1卷积生成通道注意力图。
"""

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        # 自适应平均池化，将每个通道的空间信息压缩为一个数（形状为 N×C×1×1）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 自适应最大池化，同上，但取最大值
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 两层卷积用于提取通道特征（全连接层结构）
        self.fc = nn.Sequential(
            # 第1层卷积：降维（即通道数减少）
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            # 第2层卷积：恢复原通道数
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

        # Sigmoid函数，将通道注意力压缩在0~1之间
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 对输入特征图做平均池化并通过fc
        avg_out = self.fc(self.avg_pool(x))
        # 对输入特征图做最大池化并通过fc
        max_out = self.fc(self.max_pool(x))
        # 将两者相加作为融合注意力
        out = avg_out + max_out
        # 返回加权后的注意力图（通道注意力）
        return self.sigmoid(out)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()

        # 卷积用于融合通道后的空间特征（输入通道为2：平均图+最大图）
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 对通道维进行平均池化，结果为 N×1×H×W
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 对通道维进行最大池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 沿通道维度拼接：得到 N×2×H×W
        x = torch.cat([avg_out, max_out], dim=1)
        # 卷积并激活，生成空间注意力图
        x = self.conv1(x)
        return self.sigmoid(x)

class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        # 中间通道数，用于降维
        dim = int(out_channels // factor)

        # 降维卷积（1×1）
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

        # 三种不同尺度的卷积（感受野不同）
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)

        # 空间注意力模块
        self.spatial_attention = SpatialAttentionModule()

        # 通道注意力模块
        self.channel_attention = ChannelAttentionModule(dim)

        # 升维卷积（恢复到目标通道数）
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        # 拼接两个输入特征图（例如来自不同层的特征）
        x_fused = torch.cat([x1, x2], dim=1)
        # 降维处理
        x_fused = self.down(x_fused)
        res = x_fused

        # ------------------- 多尺度空间特征提取 -------------------
        # 三种卷积提取不同感受野下的特征
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)

        # 多尺度特征相加融合
        x_fused_s = x_3x3 + x_5x5 + x_7x7

        # 乘以空间注意力权重（空间维度增强）
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        # ------------------- 通道注意力提取 -------------------
        x_fused_c = self.channel_attention(x_fused)

        # 融合空间增强和通道增强后的特征，并升维输出
        x_out = self.up(res + x_fused_s * x_fused_c)
        return x_out

if __name__ == "__main__":
    C = 32
    x1 = torch.randn(1, C, 50, 50)
    x2 = torch.randn(1, C, 50, 50)
    msaa_block = FusionConv(in_channels=2 * C, out_channels=C)
    output = msaa_block(x1, x2)
    print(f"输入张量1形状: {x1.shape}")
    print(f"输入张量2形状: {x2.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")