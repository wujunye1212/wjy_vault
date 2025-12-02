import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    论文地址：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11079687
    论文题目：DRPCA-Net: Make Robust PCA Great Again for Infrared Small Target Detection（2025 一区TOP）
    中文题目：DRPCA-Net：让鲁棒主成分分析（RPCA）在红外小目标检测中重焕活力（2025 一区TOP）
    讲解视频：https://www.bilibili.com/video/BV1DzYWzSEpt/
        动态空间注意力（Dynamic Spatial Attention，DSA）：
            实际意义：①异质特征建模能力不足：不同场景背景具有显著空间异质性，而传统方法（如卷积核、空间注意力），无法适配不同区域的背景特征差异。
                    ②低对比度小目标定位精度不足：无法聚焦于低对比度目标区域，导致目标被背景淹没。
                    ③融合后特征缺乏 “场景针对性”：无法根据目标大小、位置动态调整特征权重。
            实现方式：①先对输入特征图做全局平均池化，然后生成每个样本专属的动态卷积核。
                    ②将特征图按通道求平均得到空间描述符，用动态卷积核对其卷积。
                    ③再经Sigmoid得到自适应空间注意力图，与原输入特征图做元素级乘法，实现目标区域特征强化、背景干扰抑制。 
"""

class DynamicSpatialAttention(nn.Module):
    def __init__(self, in_channels=32, kernel_size=3):
        super().__init__()
        # 保存卷积核大小
        self.kernel_size = kernel_size
        # 定义卷积核生成器，用于动态生成注意力权重的卷积核
        self.kernel_generator = nn.Sequential(
            # 自适应平均池化，将特征图压缩为1x1大小，保留通道数
            # 输出形状: [批次大小, 输入通道数, 1, 1]
            nn.AdaptiveAvgPool2d(1),
            # 1x1卷积，用于通道维度的特征转换
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            # ReLU激活函数，引入非线性
            nn.ReLU(),
            # 1x1卷积，最终生成k×k大小的卷积核参数
            # 输出形状: [批次大小, k*k, 1, 1]，其中k是卷积核大小
            nn.Conv2d(in_channels, kernel_size ** 2, kernel_size=1)
        )
        # Sigmoid激活函数，用于将注意力权重归一化到0-1之间
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取输入特征图的形状信息
        # B: 批次大小, C: 通道数, H: 高度, W: 宽度
        B, C, H, W = x.shape

        # 使用卷积核生成器生成卷积核参数，并调整形状
        # 输出形状变为: [B, 1, kernel_size, kernel_size]
        kernels = self.kernel_generator(x).view(B, 1, self.kernel_size, self.kernel_size)

        # 对输入特征图在通道维度上求平均值，得到单通道特征图
        # 这一步是为了聚焦于空间信息，忽略通道间的差异
        x_mean = x.mean(dim=1, keepdim=True)
        # 调整形状以适应后续的分组卷积操作
        # 输出形状: [1, B, H, W]
        x_mean = x_mean.view(1, B, H, W)

        # 使用生成的动态卷积核对平均特征图进行卷积操作
        # 这里使用分组卷积，每个样本使用自己生成的卷积核
        # padding设置为kernel_size//2，保持特征图大小不变
        att = F.conv2d(
            x_mean,  # 输入特征图
            weight=kernels,  # 动态生成的卷积核
            padding=self.kernel_size // 2,  # 填充大小
            groups=B  # 分组数等于批次大小，实现每个样本独立卷积
        )

        # 调整注意力图的形状，与输入特征图的空间维度匹配
        att = att.view(B, 1, H, W)
        # 通过Sigmoid函数将注意力权重归一化到0-1范围
        att = self.sigmoid(att)
        # 将输入特征图与注意力权重相乘，实现特征的动态加权
        return x * att

if __name__ == "__main__":
    # 创建一个随机输入张量，形状为[1, 32, 50, 50]
    x = torch.randn(1, 32, 50, 50)
    # 实例化动态空间注意力模块，输入通道数为32
    model = DynamicSpatialAttention(in_channels=32)
    # 将输入张量传入模型，得到输出
    output = model(x)
    print(f"输入张量2形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")