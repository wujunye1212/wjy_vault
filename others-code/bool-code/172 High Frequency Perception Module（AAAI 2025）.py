import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT

"""
    论文地址：https://arxiv.org/abs/2412.10116
    论文题目：HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection（AAAI 2025）
    中文题目：HS-FPN：用于小目标检测的高频与空间感知特征金字塔网络 （AAAI 2025）
    讲解视频：https://www.bilibili.com/video/BV1N285zVEfz/
        基于离散余弦变换的高频感知模块（High Frequency Perception Module，HFP）：
            实际意义：①小目标特征有限：小目标像素尺寸极小，在经过骨干网络的多次下采样后，在最终特征图中仅能通过极少数像素表示，导致特征响应微弱，难以支撑精确定位和检测。
                    ②小目标特征缺乏针对性关注：传统 FPN 对各层特征采用相同处理流程，未对小目标的弱特征进行特殊增强，小目标特征易被背景噪声干扰。
            实现方式：①高频特征提取：用高通滤波器对输入特征图处理，过滤低频背景成分，保留小目标边缘、纹理等高频细节，生成高频响应图。
                    ②双通道注意力增强：1、通道路径：对高频响应图做全局池化，通过卷积生成通道权重，突出小目标特征的通道。2、空间路径：通过卷积生成空间掩码，聚焦小目标所在区域。
"""

# 定义DCT空间交互模块
class DctSpatialInteraction(nn.Module):
    def __init__(self,
                 in_channels,  # 输入特征图的通道数
                 ratio,  # 用于计算高频保留比例的参数
                 isdct=True):  # 标记是否使用DCT变换，True时在p1&p2中使用，False时在p3&p4中使用
        # 调用父类nn.Module的初始化方法
        super(DctSpatialInteraction, self).__init__()
        self.ratio = ratio
        self.isdct = isdct

        # 如果不使用DCT，创建1x1卷积用于空间注意力
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
                # 1x1卷积将输入通道数转为1，用于生成空间注意力图
                *[nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)]
            )

    # 计算权重矩阵的方法
    def _compute_weight(self, h, w, ratio):
        # 根据比例计算低频区域的高度和宽度
        h0 = int(h * ratio[0])  # 高度方向的低频区域比例
        w0 = int(w * ratio[1])  # 宽度方向的低频区域比例

        # 创建全为1的权重矩阵，大小与输入特征图的空间维度相同
        weight = torch.ones((h, w), requires_grad=False)  # 不需要计算梯度

        # 将低频区域（左上角）的权重设为0，实现过滤低频特征的效果
        weight[:h0, :w0] = 0
        return weight

    def forward(self, x):  # x是输入特征图
        # 获取输入特征图的形状：batch_size, channels, height, width
        _, _, h0, w0 = x.size()

        # 如果不使用DCT，直接通过1x1卷积生成空间注意力并与输入相乘
        if not self.isdct:
            # 用sigmoid将卷积结果归一化到0-1，作为注意力权重
            return x * torch.sigmoid(self.spatial1x1(x))

        # 对输入特征图进行二维DCT变换
        idct = DCT.dct_2d(x, norm='ortho')  # 使用正交归一化 二维离散余弦变换

        # 计算权重矩阵并移动到与输入相同的设备（CPU/GPU）
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)

        # 调整权重形状并扩展到与DCT结果相同的形状
        weight = weight.view(1, h0, w0).expand_as(idct)

        # 用权重过滤低频特征（保留高频特征）
        dct = idct * weight  # 过滤掉低频特征

        # 对处理后的DCT结果进行逆DCT变换，生成空间掩码
        dct_ = DCT.idct_2d(dct, norm='ortho')

        # 将输入特征图与生成的空间掩码相乘
        return x * dct_


# 定义DCT通道交互模块
class DctChannelInteraction(nn.Module):
    def __init__(self,
                 in_channels,  # 输入特征图的通道数
                 patch,  # 用于池化的补丁大小
                 ratio,  # 用于计算高频保留比例的参数
                 isdct=True  # 标记是否使用DCT变换
                 ):
        super(DctChannelInteraction, self).__init__()
        self.in_channels = in_channels
        self.h = patch[0]  # 补丁的高度
        self.w = patch[1]  # 补丁的宽度
        self.ratio = ratio
        self.isdct = isdct

        # 1x1卷积，用于通道注意力计算，使用分组卷积（32组）
        self.channel1x1 = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, 1, groups=32)],
        )
        self.channel2x1 = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, 1, groups=32)],
        )
        self.relu = nn.ReLU()  # ReLU激活函数

    # 计算权重矩阵的方法（与空间交互模块中的实现相同）
    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0  # 将低频区域权重设为0
        return weight

    # 前向传播方法
    def forward(self, x):  # x是输入特征图
        # 获取输入特征图的形状：batch_size, channels, height, width
        n, c, h, w = x.size()

        # 如果不使用DCT，使用普通的通道注意力机制
        if not self.isdct:  # true时在p1&p2中使用，false时在p3&p4中使用
            # 对输入进行自适应最大池化，得到1x1的特征图
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            # 对输入进行自适应平均池化，得到1x1的特征图
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))

            # 将最大池化和平均池化的结果通过ReLU激活后，再通过1x1卷积，最后相加
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))

            # 生成通道注意力权重并与输入相乘
            return x * torch.sigmoid(self.channel2x1(channel))

        # 如果使用DCT，先对输入进行二维DCT变换
        idct = DCT.dct_2d(x, norm='ortho')

        # 计算权重矩阵并移动到与输入相同的设备
        weight = self._compute_weight(h, w, self.ratio).to(x.device)

        # 调整权重形状并扩展到与DCT结果相同的形状
        weight = weight.view(1, h, w).expand_as(idct)

        # 过滤低频特征，保留高频特征
        dct = idct * weight  # 过滤掉低频特征

        # 对处理后的DCT结果进行逆DCT变换
        dct_ = DCT.idct_2d(dct, norm='ortho')

        # 对逆DCT结果进行自适应最大池化和平均池化，输出大小为patch大小
        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))

        # 对池化结果应用ReLU激活，然后在空间维度上求和，调整形状为(batch_size, channels, 1, 1)
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        # 计算通道注意力
        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)

        # 生成通道注意力权重并与输入相乘
        return x * torch.sigmoid(self.channel2x1(channel))

# 定义高频感知模块
class High_Frequency_Perception_Module(nn.Module):
    def __init__(self,
                 in_channels,  # 输入特征图的通道数
                 ratio=(0.25, 0.25),  # 高频保留比例，默认保留75%的高频区域
                 patch=(8, 8),  # 池化补丁大小
                 isdct=True):  # 是否使用DCT变换
        super(High_Frequency_Perception_Module, self).__init__()

        # 创建空间交互子模块
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        # 创建通道交互子模块
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)

        # 输出处理模块：3x3卷积（保持空间大小）+ 分组归一化
        self.out = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
              nn.GroupNorm(32, in_channels)]  # 使用32组的分组归一化
        )
    def forward(self, x):  # x是输入特征图
        # 通过空间交互模块得到空间注意力加权后的特征
        spatial = self.spatial(x)
        # 通过通道交互模块得到通道注意力加权后的特征
        channel = self.channel(x)
        # 将空间和通道注意力的结果相加，再通过输出处理模块
        return self.out(spatial + channel)

if __name__ == '__main__':
    input = torch.randn(1, 32, 50, 50)
    # 实例化高频感知模块，输入通道数为32
    model = High_Frequency_Perception_Module(in_channels=32)
    output = model(input)
    print(f"输入张量形状: {input.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")