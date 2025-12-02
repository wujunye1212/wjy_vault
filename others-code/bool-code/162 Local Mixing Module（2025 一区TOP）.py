import torch
import torch.nn as nn
from torch.nn import functional as F

"""    
    论文地址：https://ieeexplore.ieee.org/abstract/document/10817647/
    论文题目：STMNet: Single-Temporal Mask-based Network for Self-Supervised Hyperspectral Change Detection （2025 一区TOP）
    中文题目：STMNet：基于单时相掩膜的自监督高光谱变化检测网络（2025 一区TOP）
    讲解视频：https://www.bilibili.com/video/BV1pzMLznEDV/
        局部混合模块（Local Mixing Module，LMM）：
            实际意义：①局部特征融合不足：传统卷积难以捕捉多方向局部细节（如边缘、纹理）。
                    ②冗余信息干扰：HSI光谱波段中存在噪声和无关信息，传统方法无法动态抑制冗余。
                    ③小目标与边缘检测弱：小尺度变化（如单棵植被、建筑边缘）因感受野限制易被漏检，需精细局部特征提取能力。
            实现方式：①多方向卷积：用3×7和7×3卷积提取水平、垂直方向局部特征，与原始特征相加，扩大感受野并保留多方向细节。
                    ②通道注意力：通过全局平均池化压缩空间维度，经 MLP和Swish激活生成通道权重，动态抑制冗余。
                    ③特征加权融合：将多方向卷积特征与通道权重相乘，聚焦局部空间特征。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LMM(nn.Module):
    def __init__(self, channels):

        super(LMM, self).__init__()
        self.channels = channels
        dim = self.channels  # 特征维度，与通道数相同

        # 水平方向的卷积操作 - 使用3x7的长卷积核，保持空间尺寸不变
        # groups=dim表示使用深度可分离卷积，每个通道单独处理
        self.fc_h = nn.Conv2d(dim, dim, (3, 7), stride=1, padding=(1, 7 // 2), groups=dim, bias=False)
        # 垂直方向的卷积操作 - 使用7x3的长卷积核，保持空间尺寸不变
        self.fc_w = nn.Conv2d(dim, dim, (7, 3), stride=1, padding=(7 // 2, 1), groups=dim, bias=False)

        # 注意力权重生成网络，将特征映射到三个注意力系数
        self.reweight = Mlp(dim, dim // 2, dim * 3)

    def swish(self, x):
        """
        Swish激活函数实现
        """
        return x * torch.sigmoid(x)

    def forward(self, x):
        N, C, H, W = x.shape
        # 水平方向特征提取-使用水平长卷积核
        x_w = self.fc_h(x)
        # 垂直方向特征提取-使用垂直长卷积核
        x_h = self.fc_w(x)
        # 将水平特征、垂直特征和原始特征相加
        x_add = x_h + x_w + x

        # 全局平均池化，将空间维度压缩为1x1，保留通道信息
        att = F.adaptive_avg_pool2d(x_add, output_size=1)
        # 通过多层感知机生成注意力权重，将特征映射到3个通道组
        att = self.reweight(att).reshape(N, C, 3).permute(2, 0, 1)
        # 应用Swish激活函数处理注意力权重
        att = self.swish(att).unsqueeze(-1).unsqueeze(-1)

        # 根据注意力权重对三个特征进行加权融合
        x_att = x_h * att[0] + x_w * att[1] + x * att[2]
        return x_att

if __name__ == '__main__':
    input= torch.randn(1, 32, 50, 50)
    model = LMM(channels=32)
    output = model(input)
    print(f"输入张量形状: {input.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")