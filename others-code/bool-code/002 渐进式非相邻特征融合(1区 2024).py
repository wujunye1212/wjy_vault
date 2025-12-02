from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：https://arxiv.org/pdf/2306.15988
# 题目：Asymptotic Feature Pyramid Network for Labeling Pixels and Regions（2024-1区）
# 代码讲解：https://www.bilibili.com/video/BV1EHtUecEft/
# 中文：用于标记像素和区域的渐近特征金字塔网络（2024-1区）

def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    # 如果没有指定填充(pad)，则根据卷积核的大小计算默认的填充
    if not pad:
        # 当卷积核大小为奇数时，计算两边对称的填充值
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        # 如果指定了填充，则使用指定的填充值
        pad = pad

    # 使用nn.Sequential创建一个有序字典形式的序列模型
    return nn.Sequential(OrderedDict([
        # 卷积层
        ("conv", nn.Conv2d(in_channels=filter_in, out_channels=filter_out,
                           kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        # 批量归一化层
        ("bn", nn.BatchNorm2d(num_features=filter_out)),
        # ReLU激活函数
        ("relu", nn.ReLU(inplace=True)),
    ]))

class BasicBlock(nn.Module):
    def __init__(self, filter_in, filter_out):
        # 初始化父类 nn.Module
        super(BasicBlock, self).__init__()
        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=filter_in, out_channels=filter_out, kernel_size=3, padding=1)
        # 定义第一个批量归一化层
        self.bn1 = nn.BatchNorm2d(num_features=filter_out, momentum=0.1)
        # 定义 ReLU 激活函数
        self.relu = nn.ReLU(inplace=True)
        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=filter_out, out_channels=filter_out, kernel_size=3, padding=1)
        # 定义第二个批量归一化层
        self.bn2 = nn.BatchNorm2d(num_features=filter_out, momentum=0.1)

    def forward(self, x):
        # 保存输入作为残差连接
        residual = x
        # 第一层卷积 + 批量归一化 + ReLU 激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 第二层卷积 + 批量归一化
        out = self.conv2(out)
        out = self.bn2(out)
        # 残差连接
        out += residual
        # 最后一次 ReLU 激活
        out = self.relu(out)
        # 返回最终输出
        return out

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        # 定义上采样层，先进行1x1卷积调整通道数，再进行双线性插值上采样
        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),  # 1x1卷积
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')  # 上采样
        )

    def forward(self, x):
        # 通过上采样层
        x = self.upsample(x)

        return x

class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x2, self).__init__()

        # 定义下采样层，使用步长为2的卷积实现2倍下采样
        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 2, 2, 0)  # 2x2卷积，步长2
        )

    def forward(self, x):
        # 通过下采样层
        x = self.downsample(x)

        return x

class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x4, self).__init__()

        # 定义下采样层，使用步长为4的卷积实现4倍下采样
        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 4, 4, 0)  # 4x4卷积，步长4
        )

    def forward(self, x):
        # 通过下采样层
        x = self.downsample(x)

        return x

class Downsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x8, self).__init__()

        # 定义下采样层，使用步长为8的卷积实现8倍下采样
        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 8, 8, 0)  # 8x8卷积，步长8
        )

    def forward(self, x):
        # 通过下采样层
        x = self.downsample(x)

        return x

class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        # 定义两个压缩卷积层
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)

        # 定义权重融合层
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        # 定义融合后的卷积层
        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2):
        # 对两个输入特征图进行压缩
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        # 将压缩后的特征图拼接
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)

        # 计算每个级别的权重
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        # 根据权重融合特征图
        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        # 通过卷积层进一步处理融合后的特征图
        out = self.conv(fused_out_reduced)

        return out

class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        # 定义三个压缩卷积层
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)

        # 定义权重融合层
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        # 定义融合后的卷积层
        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2, input3):
        # 对三个输入特征图进行压缩
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        # 将压缩后的特征图拼接
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)

        # 计算每个级别的权重
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        # 根据权重融合特征图
        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]

        # 通过卷积层进一步处理融合后的特征图
        out = self.conv(fused_out_reduced)

        return out

class ASFF_4(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_4, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        # 定义四个压缩卷积层
        self.weight_level_0 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)

        # 定义权重融合层
        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)

        # 定义融合后的卷积层
        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input0, input1, input2, input3):
        # 对四个输入特征图进行压缩
        level_0_weight_v = self.weight_level_0(input0)
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        # 将压缩后的特征图拼接
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)

        # 计算每个级别的权重
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        # 根据权重融合特征图
        fused_out_reduced = input0 * levels_weight[:, 0:1, :, :] + \
                            input1 * levels_weight[:, 1:2, :, :] + \
                            input2 * levels_weight[:, 2:3, :, :] + \
                            input3 * levels_weight[:, 3:, :, :]

        # 通过卷积层进一步处理融合后的特征图
        out = self.conv(fused_out_reduced)

        return out

import torch.nn as nn

class BlockBody(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        # 初始化父类 nn.Module
        super(BlockBody, self).__init__()

        # 定义四个1x1卷积层，用于通道调整
        self.blocks_scalezero1 = nn.Sequential(
            BasicConv(channels[0], channels[0], 1),
        )
        self.blocks_scaleone1 = nn.Sequential(
            BasicConv(channels[1], channels[1], 1),
        )
        self.blocks_scaletwo1 = nn.Sequential(
            BasicConv(channels[2], channels[2], 1),
        )
        self.blocks_scalethree1 = nn.Sequential(
            BasicConv(channels[3], channels[3], 1),
        )

        # 定义2倍下采样和2倍上采样操作
        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        # 定义两个特征融合模块 ASFF_2
        self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])

        # 定义两个残差块序列
        self.blocks_scalezero2 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone2 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )

        # 定义多个上采样和下采样操作
        self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        # 定义三个特征融合模块 ASFF_3
        self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])

        # 定义三个残差块序列
        self.blocks_scalezero3 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone3 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo3 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )

        # 定义多个上采样和下采样操作
        self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scalezero3_8 = Downsample_x8(channels[0], channels[3])
        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
        self.downsample_scaleone3_4 = Downsample_x4(channels[1], channels[3])
        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.downsample_scaletwo3_2 = Downsample_x2(channels[2], channels[3])
        self.upsample_scalethree3_8 = Upsample(channels[3], channels[0], scale_factor=8)
        self.upsample_scalethree3_4 = Upsample(channels[3], channels[1], scale_factor=4)
        self.upsample_scalethree3_2 = Upsample(channels[3], channels[2], scale_factor=2)

        # 定义四个特征融合模块 ASFF_4
        self.asff_scalezero3 = ASFF_4(inter_dim=channels[0])
        self.asff_scaleone3 = ASFF_4(inter_dim=channels[1])
        self.asff_scaletwo3 = ASFF_4(inter_dim=channels[2])
        self.asff_scalethree3 = ASFF_4(inter_dim=channels[3])

        # 定义四个残差块序列
        self.blocks_scalezero4 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone4 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo4 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )
        self.blocks_scalethree4 = nn.Sequential(
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
        )

    def forward(self, x):
        # 解包输入特征图
        x0, x1, x2, x3 = x

        # 通过1x1卷积层调整通道
        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        x2 = self.blocks_scaletwo1(x2)
        x3 = self.blocks_scalethree1(x3)

        # 特征融合：ASFF_2
        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)

        # 通过残差块序列
        x0 = self.blocks_scalezero2(scalezero)
        x1 = self.blocks_scaleone2(scaleone)

        # 特征融合：ASFF_3
        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)

        # 通过残差块序列
        x0 = self.blocks_scalezero3(scalezero)
        x1 = self.blocks_scaleone3(scaleone)
        x2 = self.blocks_scaletwo3(scaletwo)

        # 特征融合：ASFF_4
        scalezero = self.asff_scalezero3(x0, self.upsample_scaleone3_2(x1), self.upsample_scaletwo3_4(x2), self.upsample_scalethree3_8(x3))
        scaleone = self.asff_scaleone3(self.downsample_scalezero3_2(x0), x1, self.upsample_scaletwo3_2(x2), self.upsample_scalethree3_4(x3))
        scaletwo = self.asff_scaletwo3(self.downsample_scalezero3_4(x0), self.downsample_scaleone3_2(x1), x2, self.upsample_scalethree3_2(x3))
        scalethree = self.asff_scalethree3(self.downsample_scalezero3_8(x0), self.downsample_scaleone3_4(x1), self.downsample_scaletwo3_2(x2), x3)

        # 通过残差块序列
        scalezero = self.blocks_scalezero4(scalezero)
        scaleone = self.blocks_scaleone4(scaleone)
        scaletwo = self.blocks_scaletwo4(scaletwo)
        scalethree = self.blocks_scalethree4(scalethree)

        # 返回最终融合后的特征图
        return scalezero, scaleone, scaletwo, scalethree

class AFPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],  # 输入特征图的通道数列表
                 out_channels=256):  # 输出特征图的通道数
        super(AFPN, self).__init__()

        # 设置是否使用半精度浮点数 fp16
        self.fp16_enabled = False

        # 定义1x1卷积层，用于调整输入特征图的通道数
        self.conv0 = BasicConv(in_channels[0], in_channels[0] // 8, 1)
        self.conv1 = BasicConv(in_channels[1], in_channels[1] // 8, 1)
        self.conv2 = BasicConv(in_channels[2], in_channels[2] // 8, 1)
        self.conv3 = BasicConv(in_channels[3], in_channels[3] // 8, 1)

        # 定义 BlockBody 模块，用于多尺度特征融合
        self.body = nn.Sequential(
            BlockBody([in_channels[0] // 8, in_channels[1] // 8, in_channels[2] // 8, in_channels[3] // 8])
        )

        # 定义1x1卷积层，用于调整输出特征图的通道数到统一的 `out_channels`
        # in_channels[0] // 8 的目的是为了减少输入特征图的通道数，从而达到降低计算复杂度、减少模型参数以及优化特征表示的目的。
        self.conv00 = BasicConv(in_channels[0] // 8, out_channels, 1)
        self.conv11 = BasicConv(in_channels[1] // 8, out_channels, 1)
        self.conv22 = BasicConv(in_channels[2] // 8, out_channels, 1)
        self.conv33 = BasicConv(in_channels[3] // 8, out_channels, 1)
        self.conv44 = nn.MaxPool2d(kernel_size=1, stride=2)  # 用于生成额外的下采样特征图

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 如果是卷积层
                nn.init.xavier_normal_(m.weight, gain=0.02)  # 使用 Xavier 正态分布初始化权重
            elif isinstance(m, nn.BatchNorm2d):  # 如果是批量归一化层
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)  # 初始化权重为正态分布
                torch.nn.init.constant_(m.bias.data, 0.0)  # 初始化偏置为常数0

    def forward(self, x):
        # 解包输入特征图
        x0, x1, x2, x3 = x

        # 通过1x1卷积层调整输入特征图的通道数
        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        # 通过 BlockBody 模块进行多尺度特征融合
        out0, out1, out2, out3 = self.body([x0, x1, x2, x3])

        # 通过1x1卷积层调整输出特征图的通道数
        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)

        # 生成额外的下采样特征图
        out4 = self.conv44(out3)

        # 返回最终的融合后的特征图
        return out0, out1, out2, out3, out4

if __name__ == "__main__":
    # 创建实例，输入通道数为 32
    block = AFPN()

    # 创建随机输入张量 1，形状为 [batch_size, channels, height, width]
    input1 = torch.rand(16, 256, 200, 200)
    # 创建随机输入张量 2，形状与输入张量 1 相同
    input2 = torch.rand(16, 512, 100, 100)
    # 创建随机输入张量 3，形状为 [batch_size, channels, height, width]
    input3 = torch.rand(16, 1024, 50, 50)
    # 创建随机输入张量 4，形状与输入张量 1 相同
    input4 = torch.rand(16, 2048, 25, 25)

    # 应用 Model 实例处理输入张量
    x = (input1, input2,input3, input4)
    output = block(x)
    output1, output2, output3, output4,output5 = output
    # 打印输出张量的形状
    print(output1.size())
    print(output2.size())
    print(output3.size())
    print(output4.size())
    print(output5.size())
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
