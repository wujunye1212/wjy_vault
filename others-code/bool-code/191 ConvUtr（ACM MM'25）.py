import torch
import torch.nn as nn

""" 
    论文地址：https://ieeexplore.ieee.org/abstract/document/10890177/ 
    论文题目：Mobile U-ViT: Revisiting large kernel and U-shaped ViT for efficient medical image segmentation (ACM MM'25) 
    中文题目：Mobile U-ViT：面向高效医学图像分割的轻量级混合网络 (ACM MM'25) 
    讲解视频：https://www.bilibili.com/video/BV1afCsBwEwK/
    类似Transformer大核卷积块（ConvUtr）
        实际意义：①感受野不足，难以提取医学图像中稀疏且分散的语义信息的问题：医疗图像（如 CT、超声）与自然图像不同，局部特征稀疏，重要信息分布分散，
                而传统小卷积核（3×3）的感受野有限。
                ②轻量模型表示能力不足，无法学习Transformer式的全局关系：轻量CNN的另一个劣势是通道特征交互不足，难以像 ViT 那样构建长距离依赖关系。
        实现方式：ConvUtr = 大卷积核（全局建模） +  通道间交互 + 残差结构（稳定训练）
"""

class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn  # 保存被包装的子模块

    def forward(self, x):
        return self.fn(x) + x  # 残差连接：输出 = 子模块输出 + 原输入

class ConvUtr(nn.Module):
    """
    ConvUtr：轻量卷积残差块堆叠 + 卷积上投影
    - depth：重复残差单元的次数
    - kernel：深度可分离卷积的核大小（通常取 3/5/7）【可以自定义~~】
    - ch_in -> ch_out：最后用 3x3 卷积映射到目标通道
    """
    def __init__(self, ch_in: int, ch_out: int, depth: int = 1, kernel: int = 9):
        super(ConvUtr, self).__init__()

        # 堆叠 depth 个残差单元，每个单元包含：
        # 1) 深度可分离卷积（DWConv）分支：捕获局部空间信息，计算量低
        # 2) 1x1 瓶颈 MLP 卷积分支：先升维到 4*ch_in，再降回 ch_in，增强通道表达
        self.block = nn.Sequential(
            *[nn.Sequential(
                # 分支 A：DWConv -> GELU -> BN，并用 Residual 包一层
                Residual(nn.Sequential(
                    nn.Conv2d(
                        ch_in, ch_in,
                        kernel_size=(kernel, kernel),
                        groups=ch_in,                        # groups=通道数 => 深度可分离
                        padding=(kernel // 2, kernel // 2)   # 保持特征大小不变
                    ),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                # 分支 B：1x1 升降维瓶颈 -> GELU -> BN，用 Residual 包一层
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in * 4, kernel_size=1),  # 升维
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in * 4),
                    nn.Conv2d(ch_in * 4, ch_in, kernel_size=1),  # 降回原通道
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
            ) for _ in range(depth)]
        )

        # 上投影（通道映射）：3x3 Conv + BN + ReLU，把 ch_in 映射到 ch_out
        self.up = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)  # 经过若干残差单元堆叠，增强局部与通道特征
        x = self.up(x)     # 最后映射到目标通道数，便于后续拼接/上采样
        return x

if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 50, 50)
    model = ConvUtr(ch_in=32, ch_out=64)
    output = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")