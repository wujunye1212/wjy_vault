import torch
import torch.nn as nn

"""    
    论文地址：https://arxiv.org/pdf/2504.20670
    论文题目：FBRT-YOLO: Faster and Better for Real-Time Aerial Image Detection （AAAI 2025）
    中文题目：FBRT-YOLO：面向实时航拍图像检测的更快更好方案（AAAI 2025）
    讲解视频：https://www.bilibili.com/video/BV1n4TYzbEKe/
        多核感知卷积单元（Multi-Kernel Perception Unit，MKP）：
            实际意义：①在卷积神经网络的特征提取过程中，小目标容易因下采样或卷积操作导致特征信息丢失，消失在背景中。
                    ②目标尺度差异大（如远处小目标与近处大目标），传统单尺度卷积核难以同时有效处理不同尺度的目标，导致网络对多尺度特征的感知能力有限。
            实现方式：①集成不同尺寸卷积核。
                    ②引入逐点卷积（Point-wise Convolution），实现跨尺度特征的融合与交互。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

def calculate_padding(kernel_size, padding=None, dilation=1):
    """计算填充大小，使卷积输出保持相同尺寸"""
    if dilation > 1:
        # 计算膨胀后的实际核大小
        if isinstance(kernel_size, int):
            effective_kernel = dilation * (kernel_size - 1) + 1
        else:
            effective_kernel = [dilation * (k - 1) + 1 for k in kernel_size]
    else:
        effective_kernel = kernel_size

    # 自动计算填充大小，使输入输出尺寸相同
    if padding is None:
        if isinstance(effective_kernel, int):
            padding = effective_kernel // 2
        else:
            padding = [k // 2 for k in effective_kernel]
    return padding

class ConvolutionLayer(nn.Module):
    """标准卷积层，包含卷积、批归一化和激活函数"""
    default_activation = nn.SiLU()  # 默认激活函数

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None,
                 groups=1, dilation=1, activation=True):
        """初始化卷积层"""
        super().__init__()
        # 创建卷积层，自动计算填充
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            calculate_padding(kernel_size, padding, dilation),
            groups=groups, dilation=dilation, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # 设置激活函数
        if activation is True:
            self.activation = self.default_activation
        elif isinstance(activation, nn.Module):
            self.activation = activation
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        """前向传播：卷积 -> 批归一化 -> 激活"""
        return self.activation(self.batch_norm(self.convolution(x)))

    def forward_fused(self, x):
        """前向传播（融合版本）：卷积 -> 激活"""
        return self.activation(self.convolution(x))

class MKPConv(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, activation=True):
        """初始化多尺度并行卷积模块"""
        super().__init__()

        # 3x3深度可分离卷积分支
        self.depthwise_conv_3x3 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels
        )
        self.pointwise_conv_3x3 = ConvolutionLayer(channels, channels, kernel_size=1, stride=1)

        # 5x5深度可分离卷积分支
        self.depthwise_conv_5x5 = nn.Conv2d(
            channels, channels, kernel_size=5, stride=1, padding=2, groups=channels
        )
        self.pointwise_conv_5x5 = ConvolutionLayer(channels, channels, kernel_size=1, stride=1)

        # 7x7深度可分离卷积分支
        self.depthwise_conv_7x7 = nn.Conv2d(
            channels, channels, kernel_size=7, stride=1, padding=3, groups=channels
        )

        self.pointwise_conv_7x7 = ConvolutionLayer(channels, channels, kernel_size=1, stride=1)

    def forward(self, x):
        """前向传播：多尺度卷积后残差连接"""
        x = self.depthwise_conv_3x3(x)
        x = self.pointwise_conv_3x3(x)
        x = self.depthwise_conv_5x5(x)
        x = self.pointwise_conv_5x5(x)
        x = self.depthwise_conv_7x7(x)
        x = self.pointwise_conv_7x7(x)
        return x

if __name__ == "__main__":
    x = torch.randn(1,32,50, 50)
    model = MKPConv(32)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")