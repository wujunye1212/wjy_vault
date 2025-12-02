import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2412.20066
    论文题目：MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration（CVPR 2025）
    中文题目：MaIR：一种保持局部性和连续性的用于图像恢复的Mamba模型（CVPR 2025）
    讲解视频：https://www.bilibili.com/video/BV1H7Viz9Er6/
        序列洗牌/混淆注意力（Sequence Shuffle Attention，SSA）：
            实际意义：①忽视序列间差异：现有方法在将2D图像展平为1D序列，采用像素级求和聚合序列。忽略了不同展开方式间的显著差异，未能充分挖掘不同序列的互补信息。
                    ②难以捕捉复杂依赖关系：由于图像的复杂性，不同序列之间存在着复杂的依赖关系。
            实现方式：残差网络+权重机制+特征重排
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class SequenceShuffleAttention(nn.Module):
    def __init__(self, in_features, out_features, group=4,input_resolution=(64,64)):
        super().__init__()
        # 保存分组数量
        self.group = group
        # 保存输入图像的分辨率
        self.input_resolution = input_resolution
        # 保存输入特征的数量
        self.in_features = in_features
        # 保存输出特征的数量
        self.out_features = out_features

        # 定义一个门控机制，使用nn.Sequential将多个层按顺序组合在一起
        self.gating = nn.Sequential(
            # 自适应平均池化层，将输入特征图池化为1x1的大小
            nn.AdaptiveAvgPool2d(1),
            # 卷积层，用于特征转换，使用分组卷积
            nn.Conv2d(in_features, out_features, groups=self.group, kernel_size=1, stride=1, padding=0),
            # Sigmoid激活函数，将输出值映射到0到1之间
            nn.Sigmoid()
        )

    # 定义通道洗牌函数
    def channel_shuffle(self, x):
        # 获取输入张量的批次大小、通道数、高度和宽度
        batchsize, num_channels, height, width = x.data.size()
        # 确保通道数能被分组数量整除
        assert num_channels % self.group == 0
        # 计算每个分组的通道数
        group_channels = num_channels // self.group
        # 对输入张量进行形状重塑
        x = x.reshape(batchsize, group_channels, self.group, height, width)
        # 对张量的维度进行重新排列
        x = x.permute(0, 2, 1, 3, 4)
        # 再次对张量进行形状重塑
        x = x.reshape(batchsize, num_channels, height, width)
        # 返回处理后的张量
        return x

    # 定义通道重排函数
    def channel_rearrange(self, x):
        # 获取输入张量的批次大小、通道数、高度和宽度
        batchsize, num_channels, height, width = x.data.size()
        # 确保通道数能被分组数量整除
        assert num_channels % self.group == 0
        # 计算每个分组的通道数
        group_channels = num_channels // self.group

        # 对输入张量进行形状重塑
        x = x.reshape(batchsize, self.group, group_channels, height, width)

        # 对张量的维度进行重新排列
        x = x.permute(0, 2, 1, 3, 4)
        # 再次对张量进行形状重塑
        x = x.reshape(batchsize, num_channels, height, width)
        return x

    def forward(self, x):
        residual = x
        # 对输入张量进行通道洗牌操作
        x = self.channel_shuffle(x) # (1, 32, 50, 50)
        # 通过门控机制对输入张量进行处理
        x = self.gating(x)# (1, 32, 1, 1)
        # 对处理后的张量进行通道重排操作
        x = self.channel_rearrange(x) # # (1, 32, 1, 1)
        # 将残差连接与处理后的张量相乘
        return residual * x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(1, 32, 50, 50).to(device)
    model = SequenceShuffleAttention(in_features=32, out_features=32, input_resolution=(50, 50)).to(device)
    output = model(input_tensor)
    print(f'Input size: {input_tensor.size()}')
    print(f'Output size: {output.size()}')
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")