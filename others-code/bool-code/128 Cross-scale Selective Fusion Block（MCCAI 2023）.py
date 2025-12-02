import torch.nn as nn
import torch

"""
    论文地址：https://arxiv.org/pdf/2306.14119
    论文题目：SHISRCNet: Super-resolution And Classification Network For Low-resolution Breast Cancer Histopathology Image （MICCAI 2023）
    中文题目：SHISRCNet：用于低分辨率乳腺癌组织病理学图像的超分辨率与分类网络 （MICCAI 2023）
    讲解视频：https://www.bilibili.com/video/BV1ZZQ7YmEV3/
        跨尺度选择性融合（Cross-scale Selective Fusion block ,CSFblock）：
            实际意义：①多尺度特征局限性：仅靠 FPN 直接融合高低分辨率特征存在不足。
                    ②多分辨率特征处理：低分辨率图像缺乏细节，高分辨率图像包含丰富信息。
                    ③特征选择和融合权重：传统方法在融合多尺度特征时，往往未充分考虑特征的重要性差异，导致融合效果不佳。
            实现方式：1）先把低分辨率特征放大后和高分辨率特征相加；
                    2）用相加结果生成两个智能权重；
                    3）最后用这两个权重自动调节两种特征的融合比例。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class oneConv(nn.Module):
    # 卷积层封装
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations, bias=False),
        )

    def forward(self, x):
        # 前向传播
        x = self.conv(x)
        return x

class CSFblock(nn.Module):
    # 联合网络模块
    def __init__(self, in_channels, channels_1, strides):
        super().__init__()
        self.Up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=strides, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.Fgp = nn.AdaptiveAvgPool2d(1)
        self.layer1 = nn.Sequential(
            oneConv(in_channels, channels_1, 1, 0, 1),
            oneConv(channels_1, in_channels, 1, 0, 1),
        )
        self.SE1 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE2 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x_h, x_l):
        """
        """
        """
            1、接收高分辨率特征 Xh 和低分辨率特征 Xl ，对 Xl 上采样使其与 Xh 维度一致后相加得到融合特征 U
        """
        # 高分辨率特征图
        x1 = x_h
        # 上采样低分辨率特征图
        x2 = self.Up(x_l)
        # 特征图相加
        x_f = x1 + x2

        """
            2、对U进行全局平均池化得全局信息S
        """
        # 全局平均池化
        Fgp = self.Fgp(x_f)

        """
            3、权重向量a和b，进行 softmax 操作得到归一化权重。
        """
        # 通道注意力机制
        x_se = self.layer1(Fgp)
        x_se1 = self.SE1(x_se)
        x_se2 = self.SE2(x_se)
        # 拼接通道注意力
        x_se = torch.cat([x_se1, x_se2], 2)
        # 计算注意力权重
        x_se = self.softmax(x_se)
        att_3 = torch.unsqueeze(x_se[:, :, 0], 2)
        att_5 = torch.unsqueeze(x_se[:, :, 1], 2)

        """
               4、将Xh和上采样后的Xl按通道与对应权重相乘并相加
        """
        # 加权特征图
        x1 = att_3 * x1
        x2 = att_5 * x2

        """
            5、得到最终融合特征图
        """
        x_all = x1 + x2
        return x_all

if __name__ == '__main__':
    # 假设输入通道数为64，channels_1为32，步幅为2
    in_channels = 64
    channels_1 = 32
    strides = 2
    # 创建CSFblock对象
    csf_block = CSFblock(in_channels, channels_1, strides)

    # 假设x_h和x_l的形状为(batch_size, in_channels, height, width)
    batch_size = 8
    height = 64
    width = 64
    x_h = torch.randn(batch_size, in_channels, height, width)
    x_l = torch.randn(batch_size, in_channels, height // strides, width // strides)

    # 调用CSFblock的forward方法
    output = csf_block(x_h, x_l)

    print("input1.shape:", x_h.shape)
    print("input2.shape:", x_l.shape)
    print("output.shape:", output.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
