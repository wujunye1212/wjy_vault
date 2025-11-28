import torch
import torch.nn as nn

"""
    论文地址：https://ieeexplore.ieee.org/abstract/document/10969832
    论文题目：A Lightweight Semantic Segmentation Network Based on Self-Attention Mechanism and State Space Model for Efficient Urban Scene Segmentation（2025 一区TOP）
    中文题目：基于自注意力机制和状态空间模型的轻量级语义分割网络：面向高效城市场景分割（2025 一区TOP）
    讲解视频：https://www.bilibili.com/video/BV1xBuxzREpH/
        信息整合注意力（Information Integration Attention，IIA）：
        实际意义：①空间位置信息丢失：图像经过编码器下采样后，特征图的空间分辨率降低，细节信息（如物体边缘、小目标位置）容易丢失。
                 ②噪声干扰与类别混淆：遥感图像中存在复杂场景（如阴影、相似纹理的物体），编码器与解码器的特征融合过程中容易引入噪声，导致相似类别被误判。
        实现方式：①特征拼接：融合编码器局部细节与解码器语义特征，形成基础特征矩阵。
                ②位置提取：沿高度、宽度方向池化压缩特征，用 1D 卷积生成位置注意力权重。
                ③特征加权、残差融合：通过Sigmoid激活函数调整权重，突出关键区域（如目标边缘），然后与原始特征残差连接，提升分割精准度。
"""
class AttentionWeight(nn.Module):
    def __init__(self, channel, kernel_size=7):
        # 继承自nn.Module，初始化注意力权重模块
        super(AttentionWeight, self).__init__()
        # 计算卷积填充，确保输入输出尺寸一致
        padding = (kernel_size - 1) // 2
        # 1x1卷积，将2个通道压缩为1个通道
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        # 深度可分离卷积，保持通道数不变
        self.conv2 = nn.Conv1d(channel, channel, kernel_size, padding=padding, groups=channel, bias=False)
        # 批归一化层，加速模型训练和提升稳定性
        self.bn = nn.BatchNorm1d(channel)
        # Sigmoid激活函数，将值压缩到0-1范围
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取输入张量的维度信息：批次大小、宽度、通道数、高度
        b, w, c, h = x.size()
        # 计算每个通道的最大值和平均值，沿宽度维度聚合
        x_weight = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        # 通过1x1卷积融合最大值和平均值特征，并调整形状
        x_weight = self.conv1(x_weight).view(b, c, h)
        # 应用深度可分离卷积、批归一化和Sigmoid激活
        x_weight = self.sigmoid(self.bn(self.conv2(x_weight)))
        # 重塑权重张量以匹配输入张量维度
        x_weight = x_weight.view(b, 1, c, h)
        # 将输入与注意力权重相乘，增强重要特征

        return x * x_weight

class IIA(nn.Module):
    def __init__(self, channel):
        # 继承自nn.Module，初始化IIA模块（可能代表Interleaved Inter-channel Attention）
        super(IIA, self).__init__()
        # 创建注意力权重子模块
        self.attention = AttentionWeight(channel)

    def forward(self, x):
        # 调整维度顺序，将宽度和高度维度交换，处理水平方向
        x_h = x.permute(0, 3, 1, 2).contiguous()
        # 应用注意力机制，然后恢复原始维度顺序
        x_h = self.attention(x_h).permute(0, 2, 3, 1)

        # 调整维度顺序，将通道和高度维度交换，处理垂直方向
        x_w = x.permute(0, 2, 1, 3).contiguous()
        # 应用注意力机制，然后恢复原始维度顺序
        x_w = self.attention(x_w).permute(0, 2, 1, 3)

        # 将原始输入与水平和垂直注意力增强后的特征相加，实现多尺度特征融合
        return x + x_h + x_w

if __name__ == "__main__":
    # 创建随机输入张量，批次大小为1，通道数32，宽高各50
    x = torch.randn(1, 32, 50, 50)
    # 实例化IIA模型，指定通道数
    model = IIA(32)
    # 前向传播计算输出
    output = model(x)
    # 打印输入和输出的张量形状，验证维度一致性
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    # 以下是作者信息，与模型功能无关
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")