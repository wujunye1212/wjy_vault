import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    论文地址：https://arxiv.org/abs/2403.10778
    论文题目：HCF-Net: Hierarchical Context Fusion Network for Infrared Small Object Detection（arxiv 2024）
    中文题目：HCF-Net：用于红外小物体检测的分层上下文融合网络（arxiv 2024）
    讲解视频：https://www.bilibili.com/video/BV1iZqmYFEbz/
        尺寸感知选择性集成模块（Dimension-Aware Selective Integration Module ,DASI）：
             作用：选择合适的通道数来进行特征融合。
             结构组成：自适应通道机制，根据目标的大小和特点自动调整需要使用的通道数，避免了过拟合或欠拟合的情况
"""

class Bag(nn.Module):
    def __init__(self):
        super(Bag, self).__init__()  # 初始化父类

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)  # 对输入 d 应用 sigmoid 函数，获得边缘注意力
        return edge_att * p + (1 - edge_att) * i  # 根据边缘注意力融合 p 和 i

class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1):
        super().__init__()
        # 定义卷积层
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type  # 正则化类型
        self.act = activation  # 是否使用激活函数

        if self.norm_type == 'gn':
            # 组归一化
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            # 批归一化
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # 使用 ReLU 激活函数
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)  # 应用卷积
        if self.norm_type is not None:
            x = self.norm(x)  # 应用归一化
        if self.act:
            x = self.relu(x)  # 应用激活函数
        return x

class DASI(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.bag = Bag()  # 初始化 Bag 模块
        # 定义 tail_conv 模块
        self.tail_conv = nn.Sequential(
            conv_block(in_features=out_features,
                       out_features=out_features,
                       kernel_size=(1, 1),
                       padding=(0, 0),
                       norm_type=None,
                       activation=False)
        )
        # 定义 conv 模块
        self.conv = nn.Sequential(
            conv_block(in_features=out_features // 2,
                       out_features=out_features // 4,
                       kernel_size=(1, 1),
                       padding=(0, 0),
                       norm_type=None,
                       activation=False)
        )
        self.bns = nn.BatchNorm2d(out_features)  # 批归一化

        # 定义 skips 模块
        self.skips = conv_block(in_features=in_features,
                                out_features=out_features,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                norm_type=None,
                                activation=False)
        # 定义 skips_2 模块
        self.skips_2 = conv_block(in_features=in_features * 2,
                                  out_features=out_features,
                                  kernel_size=(1, 1),
                                  padding=(0, 0),
                                  norm_type=None,
                                  activation=False)
        # 定义 skips_3 模块
        self.skips_3 = nn.Conv2d(in_features // 2, out_features,
                                 kernel_size=3, stride=2, dilation=2, padding=2)

        self.relu = nn.ReLU()  # ReLU 激活函数
        self.fc = nn.Conv2d(out_features, in_features, kernel_size=1, bias=False)  # 全连接层

        self.gelu = nn.GELU()  # GELU 激活函数

    def forward(self, x, x_low, x_high):
        if x_high is not None:
            x_high = self.skips_3(x_high)  # 通过 skips_3 处理 x_high
            x_high = torch.chunk(x_high, 4, dim=1)  # 分成 4 块

        if x_low is not None:
            x_low = self.skips_2(x_low)  # 通过 skips_2 处理 x_low
            x_low = F.interpolate(x_low, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)  # 双线性插值
            x_low = torch.chunk(x_low, 4, dim=1)  # 分成 4 块

        x_skip = self.skips(x)  # 通过 skips 处理 x
        x = self.skips(x)  # 再次通过 skips 处理 x
        x = torch.chunk(x, 4, dim=1)  # 分成 4 块

        if x_high is None:
            # 如果 x_high 为 None，融合 x 和 x_low
            x0 = self.conv(torch.cat((x[0], x_low[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_low[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_low[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_low[3]), dim=1))
        elif x_low is None:
            # 如果 x_low 为 None，融合 x 和 x_high
            x0 = self.conv(torch.cat((x[0], x_high[0]), dim=1))
            x1 = self.conv(torch.cat((x[0], x_high[1]), dim=1))
            x2 = self.conv(torch.cat((x[0], x_high[2]), dim=1))
            x3 = self.conv(torch.cat((x[0], x_high[3]), dim=1))
        else:
            # 如果两者都不为 None，使用 Bag 模块融合
            x0 = self.bag(x_low[0], x_high[0], x[0])
            x1 = self.bag(x_low[1], x_high[1], x[1])
            x2 = self.bag(x_low[2], x_high[2], x[2])
            x3 = self.bag(x_low[3], x_high[3], x[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)  # 合并 4 块
        x = self.tail_conv(x)  # 通过 tail_conv 处理
        x += x_skip  # 加上跳跃连接

        x = self.bns(x)  # 批归一化
        x = self.fc(x)  # 全连接层
        x = self.relu(x)  # ReLU 激活
        return x  # 返回输出

if __name__ == '__main__':
    batch_size = 1
    channels = 3
    height = 64
    width = 64
    x = torch.randn(batch_size, channels, height, width)
    x_low = torch.randn(batch_size, channels * 2, height // 2, width // 2)
    x_high = torch.randn(batch_size, channels // 2, height * 2, width * 2)

    # 实例化 DASI 模块
    dasinet = DASI(channels, channels * 4)

    # 打印输入和输出的形状
    output = dasinet(x, x_low, x_high)
    print("输入 x 的形状:", x.shape)
    print("输入 x_low 的形状:", x_low.shape)
    print("输入 x_high 的形状:", x_high.shape)
    print("输出的形状:", output.shape)