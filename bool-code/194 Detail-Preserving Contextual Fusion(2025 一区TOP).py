import torch
import torch.nn as nn
import torch.nn.functional as F

""" 
    论文地址：https://arxiv.org/pdf/2505.23214
    论文题目：SAMamba: Adaptive State Space Modeling with Hierarchical Vision for Infrared Small Target Detection (2025 一区TOP) 
    中文题目：SAMamba：基于层级视觉的自适应状态空间建模在红外小目标检测中的应用 (2025 一区TOP) 
    讲解视频：https://www.bilibili.com/video/BV1jJyjBSEu7/
    细节保持式上下文特征融合模块（Detail-Preserving Contextual Fusion ,DPCF）
        实际意义：①高分辨率细节在融合中被淹没的问题：在U-Net解码阶段，需要将高分辨率（细节特征）与低分辨率（语义特征）进行融合。
                但语义特征占比大，往往会压制稀疏的小目标细节，使小目标边缘、亮度对比等关键信息被覆盖，导致目标缩小、甚至完全消失，卷积方案最常见的失败原因之一。
                ②跨域泛化能力弱：传统融合方式Concat与Conv存在一个致命问题，为所有像素、所有通道使用同一融合策略。
        实现方式：自适应选择“保细节 or 加语义”，避免小目标在融合时被淹没。
"""

class AdaptiveCombiner(nn.Module):
    def __init__(self):
        super(AdaptiveCombiner, self).__init__()
        # 定义一个可学习的标量参数 d，用于控制 p 与 i 的融合比例
        self.d = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, p, i):
        # 获取输入特征图的 batch、大通道数、高和宽
        batch_size, channel, w, h = p.shape
        # 将标量 d 扩展成与输入特征相同的形状
        d = self.d.expand(batch_size, channel, w, h)
        # 使用 Sigmoid 将 d 映射到 0~1，作为融合权重
        edge_att = torch.sigmoid(d)
        # 根据注意力权重将两路特征自适应融合
        return edge_att * p + (1 - edge_att) * i


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
        # 定义基础卷积层
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        # 记录归一化类型与激活开关
        self.norm_type = norm_type
        self.act = activation

        # 根据选择使用 BN 或 GN
        if self.norm_type == 'gn':
            # 分组数不能超过通道数
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)

        # 根据开关选择激活函数
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # 卷积操作
        x = self.conv(x)
        # 执行归一化
        if self.norm_type is not None:
            x = self.norm(x)
        # 执行激活函数
        if self.act:
            x = self.relu(x)
        return x


class DPCF(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 自适应融合模块
        self.ac = AdaptiveCombiner()
        # 最后的 1×1 卷积用于通道整合
        self.tail_conv = nn.Sequential(
            conv_block(in_features=in_features,
                       out_features=out_features,
                       kernel_size=(1, 1),
                       padding=(0, 0))
        )

    def forward(self, x_low, x_high):
        # 获取低层特征的空间尺寸
        image_size = x_low.size(2)

        # 将低层特征按照通道维分成 4 份
        if x_low is not None:
            x_low = torch.chunk(x_low, 4, dim=1)

        # 处理高层特征：先上采样到同尺寸，再按通道分成 4 份
        if x_high is not None:
            x_high = F.interpolate(x_high, size=[image_size, image_size], mode='bilinear', align_corners=True)
            x_high = torch.chunk(x_high, 4, dim=1)

        # 对每一份进行自适应融合
        x0 = self.ac(x_low[0], x_high[0])
        x1 = self.ac(x_low[1], x_high[1])
        x2 = self.ac(x_low[2], x_high[2])
        x3 = self.ac(x_low[3], x_high[3])

        # 将四份重新拼接
        x = torch.cat((x0, x1, x2, x3), dim=1)
        # 使用尾部 1×1 卷积整合通道
        x = self.tail_conv(x)
        return x


if __name__ == "__main__":
    input1_tensor = torch.randn(1, 32, 50, 50)
    input2_tensor = torch.randn(1, 32, 50, 50)
    DPCF = DPCF(in_features=32, out_features=32)
    output = DPCF(input1_tensor, input2_tensor)
    print(f"输入张量形状: {input1_tensor.shape}")
    print(f"输入张量形状: {input2_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")