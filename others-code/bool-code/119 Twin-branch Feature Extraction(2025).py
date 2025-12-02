import torch
import torch.nn as nn
"""
    论文地址：https://www.sciencedirect.com/science/article/pii/S1051200425000922
    论文题目：A synergistic CNN-transformer network with pooling attention fusion for hyperspectral image classification（2025）
    中文题目：一种用于高光谱图像分类的融合池化注意力机制的协同卷积神经网络（2025）
    讲解视频：https://www.bilibili.com/video/BV1WNAdeZEWb/
        双分支特征提取（Twin-branch Feature Extraction, TBFE）：
            实际意义：①充分利用光谱和空间信息。
                    ②降低计算复杂度，利用低输出通道逐点卷积层减少计算量。
                    ③增强模型特征表示能力，融合两个分支的特征图。
            实现方式：首先，逐点卷积调整通道维度实现降低复杂度，得特征图F。接着，通过3D卷积捕捉光谱特征，2D卷积获取空间特征。
                    再通过特征融合，结合光谱依赖和空间模式的优势。最后，激活函数和归一化层，缓解梯度问题、增强泛化能力。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""
# Twin-branch Feature Extraction
class TBFE(nn.Module):
    def __init__(self, input_channels, reduction_N=32):
        """
            input_channels: 输入通道数（如RGB图像为3）
            reduction_N: 特征压缩维度（默认32）
        """
        super(TBFE, self).__init__()

        # 点卷积（通道降维）[2,4](@ref)
        # 1x1卷积核实现跨通道信息交互，类似SENet[2](@ref)的通道注意力机制
        self.point_wise = nn.Conv2d(input_channels, reduction_N,
                                    kernel_size=1, padding=0, bias=False)

        # 深度可分离卷积（空间特征提取）[2,4](@ref)
        # 3x3卷积核提取局部空间特征，BN+ReLU增强非线性表达能力
        self.depth_wise = nn.Sequential(
            nn.Conv2d(reduction_N, reduction_N, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(reduction_N),
            nn.ReLU(),
        )

        # 三维卷积（时序特征建模）[1,4](@ref)
        # 沿通道维度构建时序建模能力，kernel_size=(1,1,3)表示在通道维度进行时序卷积
        self.conv3D = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1, 3),  # (深度,高度,宽度)
            padding=(0, 0, 1),  # 保持时序维度尺寸不变
            stride=(1, 1, 1),
            bias=False
        )

        # 特征融合与恢复
        self.bn = nn.BatchNorm2d(2 * reduction_N)  # 融合双分支特征
        self.relu = nn.ReLU()

        # 投影层（恢复原始通道数）[2](@ref)
        # 1x1卷积实现通道维度变换，类似ResNet[2](@ref)的shortcut连接
        self.pro = nn.Conv2d(2 * reduction_N, input_channels,
                             kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        """前向传播过程（含维度变换注释）"""
        # 原始输入尺寸：(batch_size, input_channels, H, W)
        # 阶段1：通道压缩
        x_1 = self.point_wise(x)  # 输出尺寸：(B, reduction_N, H, W)

        # 阶段2：空间特征提取（含残差连接）[2](@ref)
        x_2 = self.depth_wise(x_1)  # 输出尺寸保持(B, reduction_N, H, W)
        x_2 = x_1 + x_2  # 残差连接，避免梯度消失

        # 阶段3：时序特征建模[1,4](@ref)
        x_3 = x_1.unsqueeze(1)  # 增加时间维度：(B, 1, reduction_N, H, W)
        x_3 = self.conv3D(x_3)  # 3D卷积处理：(B, 1, reduction_N, H, W)
        x_3 = x_3.squeeze(1)  # 压缩时间维度：(B, reduction_N, H, W)

        # 阶段4：特征融合
        x = torch.cat((x_2, x_3), dim=1)  # 通道拼接：(B, 2*reduction_N, H, W)

        x = self.bn(x)  # 标准化处理
        x = self.relu(x)  # 非线性激活
        x = self.pro(x)  # 通道恢复：(B, input_channels, H, W)
        return x

if __name__ == "__main__":
    model = TBFE(input_channels=16)  # 实例化模块（模拟处理RGB图像）
    input = torch.randn(1, 16, 128, 128)  # 输入张量：(batch=1, channel=3, height=128, width=128)
    output = model(input)

    print('input_size:', input.size())  # 打印输入尺寸
    print('output_size:', output.size())  # 打印输出尺寸
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')
    print("公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")