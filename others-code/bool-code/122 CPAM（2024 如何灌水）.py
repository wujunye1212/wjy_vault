import torch
import torch.nn as nn
import math
"""
    论文地址：https://arxiv.org/abs/2312.06458
    论文题目：ASF-YOLO: A Novel YOLO Model with Attentional Scale Sequence Fusion for Cell Instance Segmentation（2024 三区）
    中文题目：ASF-YOLO：一种用于细胞实例分割的注意力尺度序列融合的YOLO模型（2024 三区）
    讲解视频：https://www.bilibili.com/video/BV14c9FYDERr/
        通道和位置注意力机制（Channel and position attention mechanism, CPAM）：
            实际意义：①小物体特征挖掘不足：传统 SENet 通道注意力机制降维有副作用。
                     ②细胞位置信息提取不准。
            实现方式：①通道注意力：采用无降维机制，利用1D卷积挖掘小物体多通道特征。
                     ②位置注意力：在水平和垂直方向拆分特征图，保留空间结构信息，生成位置注意力坐标。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""
# 定义通道注意力模块
class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        # 计算动态卷积核大小，公式为 log2(channel) + b / gamma，并确保卷积核大小为奇数
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        # 自适应平均池化，将每个通道的空间维度压缩为 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D 卷积，用于提取通道间的关系
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        # Sigmoid 激活函数，用于生成通道权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # 自适应平均池化，得到形状为 [batch_size, channel, 1, 1]
        """不理解"""
        y = y.squeeze(-1)  # 去掉最后一个维度，形状变为 [batch_size, channel, 1]
        y = y.transpose(-1, -2)  # 转置，形状变为 [batch_size, 1, channel]
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # 1D 卷积后再转置，并增加一个维度，形状为 [batch_size, channel, 1, 1]
        y = self.sigmoid(y)  # 通过 Sigmoid 生成通道权重

        return x * y.expand_as(x)  # 将权重扩展到与输入相同的形状，并逐元素相乘

# 定义局部注意力模块
class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        # 1x1 卷积，用于通道压缩
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)
        # 激活函数 ReLU
        self.relu = nn.ReLU()
        # 批归一化层
        self.bn = nn.BatchNorm2d(channel // reduction)

        # 两个 1x1 卷积，用于生成水平和垂直方向的注意力权重
        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        # 两个 Sigmoid 激活函数，用于生成水平和垂直方向的权重
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()  # 获取输入的高度和宽度

        # 计算水平方向的全局特征，沿宽度方向求平均，形状为 [batch_size, channel, height, 1]
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)  # 转置以便后续处理
        # 计算垂直方向的全局特征，沿高度方向求平均，形状为 [batch_size, channel, 1, width]
        x_w = torch.mean(x, dim=2, keepdim=True)
        # 将水平和垂直特征拼接后通过 1x1 卷积、批归一化和 ReLU 激活
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        # 将特征分割为水平和垂直两个部分
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        # 通过 1x1 卷积和 Sigmoid 激活生成水平和垂直方向的注意力权重
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        # 将水平和垂直的权重扩展到输入形状，并逐元素相乘
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out

# 定义 CPAM 模块，结合通道注意力和局部注意力
class CPAM(nn.Module):
    def __init__(self, ch):
        """
        CPAM 模块初始化
        :param ch: 输入特征图的通道数
        """
        super().__init__()
        # 通道注意力模块
        self.channel_att = channel_att(ch)
        # 局部注意力模块
        self.local_att = local_att(ch)

    def forward(self, input1, input2):
        # 对输入 1 进行通道注意力处理
        input1 = self.channel_att(input1)
        # 将处理后的输入 1 和输入 2 相加
        x = input1 + input2
        # 对相加结果进行局部注意力处理
        x = self.local_att(x)
        return x

# 主程序入口
if __name__ == '__main__':
    # 初始化 CPAM 模型，输入通道数为 64
    model = CPAM(ch=64)
    # 定义两个输入张量，形状为 [1, 64, 32, 32]
    input1 = torch.randn(1, 64, 32, 32)  # 假设批量大小为 1，通道数为 64，空间尺寸为 32x32
    input2 = torch.randn(1, 64, 32, 32)
    # 将输入张量传入模型，得到输出
    output = model(input1, input2)

    # 打印输入和输出的形状
    print("Input 1 shape:", input1.shape)
    print("Input 2 shape:", input2.shape)
    print("Output shape:", output.shape)
    # 计算模型的总参数量，并打印
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')
    # 打印提示信息
    print("公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
