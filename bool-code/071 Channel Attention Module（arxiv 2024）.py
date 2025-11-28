from torch import nn
import math
import torch
"""
    论文地址：https://arxiv.org/abs/2403.10778
    论文题目：HCF-Net: Hierarchical Context Fusion Network for Infrared Small Object Detection（arxiv 2024）
    中文题目：HCF-Net：用于红外小物体检测的分层上下文融合网络（arxiv 2024）
    讲解视频：https://www.bilibili.com/video/BV1qmquYpEwb/
        并行化补丁感知注意模块（Parallelized Patch-Aware Attention Module，PPA）：
             作用：捕捉不同尺度和级别的特征信息，提高小物体的识别精度。
             理论支撑：多分支特征提取策略可以同时提取不同尺度的特征信息，从而更全面地了解目标形态。
             
    【注意力部分代码】
"""


class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()  # 初始化父类
        # 计算卷积核大小
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1  # 确保卷积核大小为奇数
        padding = kernel_size // 2  # 计算填充大小
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)  # 自适应平均池化到 1x1
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),  # 一维卷积
            nn.Sigmoid()  # Sigmoid 激活函数
        )

    def forward(self, x):
        out = self.pool(x)  # 对输入 x 进行自适应平均池化
        out = out.view(x.size(0), 1, x.size(1))  # 调整张量形状以适应卷积
        out = self.conv(out)  # 应用一维卷积和 Sigmoid
        out = out.view(x.size(0), x.size(1), 1, 1)  # 恢复张量形状

        return out * x  # 通道注意力乘以原始输入


if __name__ == '__main__':

    input_tensor = torch.randn(2, 16, 32, 32)
    eca_layer = ECA(in_channel=16)
    output = eca_layer(input_tensor)

    print("Output shape:", output.shape)

    # 打印社交媒体账号信息
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")