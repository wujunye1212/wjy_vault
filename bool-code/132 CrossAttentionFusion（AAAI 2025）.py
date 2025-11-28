import torch.nn as nn
import torch

"""
    论文地址：https://arxiv.org/abs/2412.09319
    论文题目：FAMNet: Frequency-aware Matching Network for Cross-domain Few-shot Medical Image Segmentation（AAAI2025）
    中文题目：FAMNet：用于跨域少样本医学图像分割的频率感知匹配网络（AAAI2025）
    讲解视频：https://www.bilibili.com/video/BV1L2XPYkEMG/
    跨注意力机制特征融合模块（Cross-Attention-based feature fusion module，CAmodule）：
        实际意义：①融合频率域特征：图像不同频率带信息不同，低频和高频含颜色、风格信息，中频含结构和形状信息。传统方法难以融合这些特征
                ②抑制域变化信息：不同域的图像在低高频部分存在显著差异，即包含域变化信息（DVI）。直接丢弃高低频特征会损失有用信息，而保留这些特征又可能引入干扰。
        实现方式：见代码。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttentionFusion, self).__init__()
        # 定义查询（query）线性层，输入和输出维度均为embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        # 定义键（key）线性层，输入和输出维度均为embed_dim
        self.key = nn.Linear(embed_dim, embed_dim)
        # 定义值（value）线性层，输入和输出维度均为embed_dim
        self.value = nn.Linear(embed_dim, embed_dim)
        # 定义softmax层，用于在最后一个维度上进行归一化操作
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q_feature, K_feature):
        # 获取查询特征Q_feature的批量大小B、序列长度N和特征维度C
        B, N, C = Q_feature.shape

        # 通过查询线性层对查询特征进行变换，输出形状为 [B, N, C]
        Q = self.query(Q_feature)

        # 通过键线性层对键特征进行变换，输出形状为 [B, N, C]
        K = self.key(K_feature)
        # 通过值线性层对键特征进行变换，输出形状为 [B, N, C]
        V = self.value(K_feature)

        # 计算注意力分数，通过查询矩阵Q与键矩阵K的转置相乘，并除以特征维度C的平方根
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(C, dtype=torch.float32))
        # 对注意力分数应用softmax函数，得到注意力权重，形状为 [B, N, N]
        attention_weights = self.softmax(attention_scores)

        # 通过注意力权重与值矩阵V相乘，得到经过注意力加权的特征，形状为 [B, N, C]
        attended_features = torch.matmul(attention_weights, V)

        return attended_features

if __name__ == '__main__':
    feature_dim = 64
    block = CrossAttentionFusion(feature_dim)
    H = 28
    W = 28
    L = H * W
    # 生成随机的低频特征张量，形状为 (1, 128, feature_dim)
    input1 = torch.rand(1, L, feature_dim)
    # 生成随机的中频特征张量，形状为 (1, 128, feature_dim)
    input2 = torch.rand(1, L, feature_dim)
    output = block(input1, input2)

    print(f"Low input size: {input1.size()}")
    print(f"Mid input size: {input2.size()}")
    print(f"Output size: {output.size()}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
