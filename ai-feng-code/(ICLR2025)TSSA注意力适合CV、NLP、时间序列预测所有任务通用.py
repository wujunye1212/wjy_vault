import torch
import torch.nn as nn
from einops import rearrange
'''
来自ICLR2025顶会  适用于CV、NLP、时间序列预测所有任务通用--即插即用注意力模块
即插即用注意力模块： TSSA 

本文主要内容;
    注意力机制可以说是 Transformer 架构的关键区分因素，而 Transformer 近年来在各种任务上都展现了最先进的性能。
然而，Transformer 的注意力机制通常会带来较大的计算开销，其计算复杂度随着 token 数量呈二次增长。
在本研究中，我们提出了一种新的 Transformer 注意力机制，其计算复杂度随着 token 数量呈线性增长。
    
    我们的方法基于先前的研究，该研究表明，通过“白盒”架构设计，Transformer 风格的架构可以自然地生成，
其中网络的每一层被设计为执行最大编码率缩减（MCR²）目标的增量优化步骤。
具体而言，我们推导出 MCR² 目标的一种新的变分形式，并证明从该变分目标的展开梯度下降中，
可以得到一个新的注意力模块——Token Statistics Self-Attention（TSSA）。
    
    TSSA 具有线性的计算和存储复杂度，并且与传统的注意力架构完全不同，后者通常通过计算 token 之间的两两相似度来实现注意力机制。
我们的实验表明，在视觉、自然语言处理以及长序列任务上，仅仅用 TSSA 替换标准的自注意力，
就能够在计算成本显著降低的情况下，实现与传统 Transformer 相当的性能。
此外，我们的结果也对传统认知提出了挑战，即 Transformer 之所以成功，是否真的依赖于基于两两相似度的注意力机制。

TSSA注意力总结:
    TSSA 模块旨在提升注意力机制对局部和全局特征的建模能力，特别是针对视觉任务中不同区域信息的重要性差异。
    它通过引入 Token 统计信息（如均值、方差等）来增强自注意力机制，使得注意力分配更加合理，从而提高特征提取的精准度和网络的整体表现。

原理：
TSSA 结合了传统自注意力机制和 Token 统计特征，主要包括以下几个关键步骤：
    1. Token 统计特征提取：计算输入特征图中Token 的统计信息，如均值和方差，以获取全局和局部的统计分布。
    2. 注意力权重计算：将统计特征与输入特征结合，通过自注意力机制计算加权注意力分布，使得模型能够更好地关注关键区域。
    3. 自适应特征增强：基于计算得到的注意力权重，调整输入特征的分布，使得重要信息得到强化，抑制冗余或无关信息。
    4. 输出优化特征：经过 TSSA 处理后的特征更具表达力，同时保留了局部结构信息和全局关系，提高了模型对复杂场景的适应性。
TSSA 主要通过 Token 统计特征的引入来优化注意力机制，使得模型能够更加精准地学习图像中的关键特征，在各种视觉任务中（如分类、检测、分割等）都有潜在的应用价值

TSSA模块适用：图像分类、目标检测、图像分割、遥感语义分割、图像增强、图像去噪、暗光增强等CV所有任务；NLP所有任务; 时间序列预测所有任务通过即插即用模块。

'''
class TSSA(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()

        self.heads = num_heads

        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)

        self.temp = nn.Parameter(torch.ones(num_heads, 1))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        b, h, N, d = w.shape

        w_normed = torch.nn.functional.normalize(w, dim=-2)
        w_sq = w_normed ** 2

        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)  # b * h * n

        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


if __name__ == "__main__":
    #创建TSSA模块实例，64代表通道维度
    TSSA = TSSA(64)

    # 1.如何输入的是图片4维数据 . CV方向的小伙伴都可以拿去使用   输入 B C H W, 输出 B C H W
    # 随机生成输入4维度张量：B, C, H, W
    input_img = torch.randn(1, 64, 32, 32)
    input1 = input_img
    input_img = input_img.reshape(1, 64, -1).transpose(-1, -2)
    # 运行前向传递
    output = TSSA(input_img)
    output = output.view(1, 64, 32, 32)  # 将三维度转化成图片四维度张量
    # 输出输入图片张量和输出图片张量的形状
    print("CV_TSSA_input size:", input1.size())
    print("CV_TSSA_output size:", output.size())

    # 2.如何输入的3维数据 . NLP或时序任务方向的小伙伴都可以拿去使用  输入 B L C, 输出 B L C
    B, N, C = 1, 1024, 64  # 批量大小、序列长度、特征维度
    input2 = torch.randn(B, N, C)
    output = TSSA(input2)
    print('NLP_TSSA_input size:',input2.size())
    print('NLP_TSSA_output size:',output.size())