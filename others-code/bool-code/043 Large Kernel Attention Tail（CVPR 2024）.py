import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    论文地址：https://arxiv.org/abs/2209.14145
    论文题目：Multi-scale Attention Network for Single Image Super-Resolution（CVPR 2024）
    中文题目：单图像超分辨率的多尺度注意力网络
    讲解视频：https://www.bilibili.com/video/BV1FDDuYtE7A/
        Large Kernel Attention Tail (LKAT) 大核注意尾
        用于解决：普通卷积层被广泛用作深度提取骨干的尾部。
                然而，它在建立长距离方面存在缺陷，因此限制了重建特征的代表性能力。
'''
"""
    本代码很简单，可以作为一个框架 进行填充~
"""
class LKAT(nn.Module):
    def __init__(self, n_feats):
        super().__init__()  # 调用父类初始化方法

        # 定义卷积层conv0，使用1x1卷积核，不改变特征图尺寸
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),  # 1x1卷积，输入输出通道数相同
            nn.GELU())  # 使用GELU激活函数

        # 定义注意力机制模块att，包含多个卷积层
        # 【填充注意力】
        # https://space.bilibili.com/346680886/search/video?keyword=%E6%B3%A8%E6%84%8F%E5%8A%9B
        self.att = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats),  # 深度可分离卷积，7x7卷积核
            nn.Conv2d(n_feats, n_feats, 9, stride=1, padding=(9 // 2) * 3,
                        groups=n_feats, dilation=3),                    # 深度可分离卷积，9x9卷积核，空洞率为3
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))  # 1x1卷积，用于降维或升维

        # 定义卷积层conv1，使用1x1卷积核，不改变特征图尺寸
        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)  # 1x1卷积，输入输出通道数相同

    def forward(self, x):
        x = self.conv0(x)  # 应用卷积层conv0
        x = x * self.att(x)  # 将特征图与注意力机制的结果相乘
        x = self.conv1(x)  # 应用卷积层conv1
        return x

if __name__ == '__main__':

    mab = LKAT(64)  # 创建LKAT模型实例
    input_tensor = torch.randn(1, 64, 128, 128)  # 生成随机输入张量
    output_tensor = mab(input_tensor)  # 将输入张量传递给模型进行前向传播

    print(f"Input Tensor Shape: {input_tensor.shape}")  # 打印输入张量的形状
    print(f"Output Tensor Shape: {output_tensor.shape}")  # 打印输出张量的形状

    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息