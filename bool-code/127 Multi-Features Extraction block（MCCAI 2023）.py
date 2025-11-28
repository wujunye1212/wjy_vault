import torch.nn as nn
import torch

"""
    论文地址：https://arxiv.org/pdf/2306.14119
    论文题目：SHISRCNet: Super-resolution And Classification Network For Low-resolution Breast Cancer Histopathology Image （MICCAI 2023）
    中文题目：SHISRCNet：用于低分辨率乳腺癌组织病理学图像的超分辨率与分类网络 （MICCAI 2023）
    讲解视频：https://www.bilibili.com/video/BV136RgYZEsF/
        多特征提取模块（Multi-Features Extraction block , MFEblock）：
            实际意义：传统算法固定感受野问题：多数单超分辨率方法只有固定感受野，无法有效捕获多尺度特征，难以处理不同放大倍数低分辨率图像中因分辨率低导致的信息丢失问题。
            实现方式：首先，对多尺度特征进行全局平均池化得到平均通道权重，再用 Sigmoid 函数将权重映射到 0 - 1，接着用Softmax操作归一化，
                    最后将各特征与对应归一化权重相乘并相加，生成融合后的多尺度特征。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class oneConv(nn.Module):
    # 卷积层加ReLU激活函数的封装
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations, bias=False)
        )

    def forward(self, x):
        # 前向传播
        x = self.conv(x)
        return x

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        # ASPP卷积模块，包含卷积、批归一化和ReLU激活
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class MFEblock(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        # 多特征提取模块初始化
        super(MFEblock, self).__init__()
        out_channels = in_channels
        rate1, rate2, rate3 = tuple(atrous_rates)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.layer2 = ASPPConv(in_channels, out_channels, rate1)
        self.layer3 = ASPPConv(in_channels, out_channels, rate2)
        self.layer4 = ASPPConv(in_channels, out_channels, rate3)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.Sigmoid = nn.Sigmoid()

        self.SE1 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE2 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE3 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE4 = oneConv(in_channels, in_channels, 1, 0, 1)

    def forward(self, x):
        # x: 输入的特征图 (B, C, H, W)

        # 多特征提取
        y0 = self.layer1(x)    # 第一个分支
        y1 = self.layer2(y0 + x) # 第二个分支
        y2 = self.layer3(y1 + x) # 第三个分支
        y3 = self.layer4(y2 + x) # 第四个分支

        # 多尺度融合
        y0_weight = self.SE1(self.gap(y0))
        y1_weight = self.SE2(self.gap(y1))
        y2_weight = self.SE3(self.gap(y2))
        y3_weight = self.SE4(self.gap(y3))

        # 拼接全局信息
        weight = torch.cat([y0_weight, y1_weight, y2_weight, y3_weight], 2)
        # 计算权重
        weight = self.softmax(self.Sigmoid(weight))

        # 调整权重维度
        y0_weight = torch.unsqueeze(weight[:, :, 0], 2)
        y1_weight = torch.unsqueeze(weight[:, :, 1], 2)
        y2_weight = torch.unsqueeze(weight[:, :, 2], 2)
        y3_weight = torch.unsqueeze(weight[:, :, 3], 2)

        # 加权求和
        x_att = y0_weight * y0 + y1_weight * y1 + y2_weight * y2 + y3_weight * y3
        return self.project(x_att + x)

if __name__ == '__main__':
    # 输入张量 (B, C, H, W)
    input = torch.rand(1, 64, 32, 32)
    # 扩张率
    Model = MFEblock(in_channels=64, atrous_rates=[2, 4, 8])
    out = Model(input)
    print("input.shape:", input.shape)
    print("output.shape:", out.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")