import torch
from torch import nn
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
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)  # 定义卷积层
        self.sigmoid = nn.Sigmoid()  # 定义 Sigmoid 激活函数

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)  # 计算通道维度上的平均值
        maxout, _ = torch.max(x, dim=1, keepdim=True)  # 计算通道维度上的最大值
        out = torch.cat([avgout, maxout], dim=1)  # 将平均值和最大值沿通道维拼接
        out = self.sigmoid(self.conv2d(out))  # 应用卷积和 Sigmoid 激活

        return out * x  # 将注意力权重乘以输入特征图


if __name__ == '__main__':
    input_tensor = torch.randn(2, 16, 32, 32)

    eca_layer = SpatialAttentionModule()
    output = eca_layer(input_tensor)

    print("Output shape:", output.shape)

    # 打印社交媒体账号信息
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")