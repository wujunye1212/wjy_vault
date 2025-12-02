import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    论文地址：https://arxiv.org/abs/2209.14145
    论文题目：Multi-scale Attention Network for Single Image Super-Resolution（CVPR 2024）
    中文题目：单图像超分辨率的多尺度注意力网络
    讲解视频：https://www.bilibili.com/video/BV1N3DbYdEmv/
        Gated Spatial Attention Unit (GSAU)（门控空间注意力单元）:
        解决问题：在Transformer中，FFN是增强特征表示的必要部分。然而，对于大图像而言，使用具有宽中间通道的多层感知器（MLP），参数量大。
        提出方案：整合Gate机制和空间注意力来消除非必要线性层并聚合有信息的空间上下文，与多层感知器（MLP）相比，其性能更好，并且减少了参数和计算。为了更有效地捕获空间信息，采用单层深度卷积来加权特征图。
'''
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 初始化权重参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # 初始化偏置参数
        self.eps = eps  # 小常数，防止除以零
        self.data_format = data_format  # 数据格式，支持'channels_last'和'channels_first'
        # 检查数据格式是否为支持的两种之一
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError  # 如果不是，则抛出异常
        self.normalized_shape = (normalized_shape,)  # 归一化形状

    def forward(self, x):
        # 根据数据格式选择不同的归一化方式
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)  # 使用PyTorch内置的层归一化
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)  # 计算均值
            s = (x - u).pow(2).mean(1, keepdim=True)  # 计算方差
            x = (x - u) / torch.sqrt(s + self.eps)  # 进行归一化
            x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 应用权重和偏置
            return x  # 返回归一化后的结果

class GSAU(nn.Module):
    def __init__(self, n_feats):
        super().__init__()  # 调用父类初始化方法
        i_feats = n_feats * 2  # 输入特征的数量是输出特征数量的两倍

        # 这里有个参数可以调解，根据自己任务进行修改 试试效果
        self.norm = LayerNorm(n_feats, data_format='channels_first')            # 创建层归一化实例
        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)                       # 第一个卷积层，用于扩展特征维度

        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)  # 深度可分离卷积
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)                         # 第二个卷积层，用于压缩特征维度

        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)  # 可学习的比例参数

    # 前向传播函数
    def forward(self, x):
        shortcut = x.clone()            # 创建输入张量的副本，用于残差连接  torch.Size([1, 64, 128, 128])

        # 【图中PWConv】
        x = self.Conv1(self.norm(x))        # 对输入进行层归一化后通过第一个卷积层 torch.Size([1, 128, 128, 128])

        a, x = torch.chunk(x, 2, dim=1)      # 将输出分为两个部分，a用于深度可分离卷积，x用于后续处理  torch.Size([1, 64, 128, 128])
        x = x * self.DWConv1(a)             # 对a应用深度可分离卷积，并与x相乘  torch.Size([1, 64, 128, 128])

        x = self.Conv2(x)                   # 通过第二个卷积层  torch.Size([1, 64, 128, 128])

        return x * self.scale + shortcut   # torch.Size([1, 64, 128, 128])   应用比例参数并添加残差连接 ===>  缩放因子与 残差组合

if __name__ == '__main__':
    n_feats = 64  # 特征数量

    input_tensor = torch.randn(1, n_feats, 128, 128)  # 生成随机输入张量

    mab = GSAU(n_feats)  # 创建GSAU实例

    output_tensor = mab(input_tensor)  # 获取模型输出

    print(f"Input Tensor Shape: {input_tensor.shape}")  # 打印输入张量的形状
    print(f"Output Tensor Shape: {output_tensor.shape}")  # 打印输出张量的形状

    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息