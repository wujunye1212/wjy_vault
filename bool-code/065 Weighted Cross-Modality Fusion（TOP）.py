import torch
import torch.nn as nn

"""
    论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0925231222003848
    论文题目：FCMNet: Frequency-aware cross-modality attention networks for RGB-D salient object detection
    中文题目：FCMNet：用于 RGB-D 显著物体检测的频率感知跨模态注意力网络
    讲解视频：https://www.bilibili.com/video/BV1GM6AYiEnj/
        加权跨模态融合（Weighted Cross-Modality Fusion, WCMF）：
           问题：多模态融合通常采用元素求和或级联，涉及冗余特征，忽略了图像内容信息，忽略了神经网络的非线性表示能力。
           快速梳理：空间频率通道注意(SFCA)从空间和频域中捕获互补信息。RGB分支和深度分支的特征图经过SFCA 模块，然后进行元素乘法以交互不同模态信息。
"""
class WCMF(nn.Module):
    def __init__(self, channel=256):
        super(WCMF, self).__init__()
        # 定义用于处理RGB输入的卷积层序列
        self.conv_r1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),  # 1x1卷积
            nn.BatchNorm2d(channel),              # 批归一化
            nn.ReLU()                             # 激活函数
        )
        # 定义用于处理深度输入的卷积层序列
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),  # 1x1卷积
            nn.BatchNorm2d(channel),              # 批归一化
            nn.ReLU()                             # 激活函数
        )
        # 定义融合特征的卷积层序列
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(2*channel, channel, 3, 1, 1),  # 3x3卷积
            nn.BatchNorm2d(channel),                 # 批归一化
            nn.ReLU()                                # 激活函数
        )
        # 定义输出权重的卷积层序列
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(channel, 2, 3, 1, 1),         # 3x3卷积
            nn.BatchNorm2d(2),                      # 批归一化
            nn.ReLU()                               # 激活函数
        )
        # 定义自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def fusion(self, f1, f2, f_vec):
        # 提取权重
        w1 = f_vec[:, 0, :, :].unsqueeze(1)  # 提取第一个通道的权重
        w2 = f_vec[:, 1, :, :].unsqueeze(1)  # 提取第二个通道的权重
        # 计算加权和
        out1 = (w1 * f1) + (w2 * f2)
        # 计算加权乘积
        out2 = (w1 * f1) * (w2 * f2)
        # 返回融合结果
        return out1 + out2

    def forward(self, rgb, depth):
        # 处理RGB输入
        Fr = self.conv_r1(rgb)
        # 处理深度输入
        Fd = self.conv_d1(depth)

        # 特征拼接
        f = torch.cat([Fr, Fd], dim=1)
        # 融合特征
        f = self.conv_c1(f)
        # 计算权重
        f = self.conv_c2(f)

        # 进行特征融合
        Fo = self.fusion(Fr, Fd, f)
        return Fo

if __name__ == '__main__':
    # 创建RGB和深度输入的假设张量
    rgb_input = torch.randn(1, 64, 32, 32)  # RGB输入
    depth_input = torch.randn(1, 64, 32, 32)  # 深度输入

    # 通过WCMF模型
    wcmf = WCMF(64)
    output = wcmf(rgb_input, depth_input)

    # 打印输入和输出的shape
    print("RGB:", rgb_input.shape)
    print("深度:", depth_input.shape)
    print("输出形状:", output.shape)

    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息
