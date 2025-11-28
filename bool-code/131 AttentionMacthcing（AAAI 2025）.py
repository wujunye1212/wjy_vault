import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://arxiv.org/abs/2412.09319
    论文题目：FAMNet: Frequency-aware Matching Network for Cross-domain Few-shot Medical Image Segmentation（AAAI2025）
    中文题目：FAMNet：用于跨域少样本医学图像分割的频率感知匹配网络（AAAI2025）
    讲解视频：https://www.bilibili.com/video/BV1miXGYHE3d/
        注意力匹配机制（Multi-Spectrum Attention-based Matching ，ABM）：
            实际意义：①避免过拟合：不同域的图像的高、低频信号差异大，传统模型依赖特定域频率训练，易过拟合，性能下降。
                    ②增强泛化能力：根据频率类型不同实现加权，学习域无关相似性。
            实现方式：1）矩阵线性变换映射联合空间，减少域内差异，增强匹配稳定性；
                    2）用余弦相似度和 sigmoid 函数算注意力矩阵，让模型学域无关相似性；
                    3）无关域突出相似部分，特定域带抑制相似部分，经 MLP 融合得到最终特征表示。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class AttentionMacthcing(nn.Module):
    def __init__(self,  seq_len=5000):
        super(AttentionMacthcing, self).__init__()
        self.fc_spt = nn.Sequential(
            # 第一个全连接层，将输入维度从seq_len降为seq_len // 10
            nn.Linear(seq_len, seq_len // 10),
            # 使用ReLU激活函数
            nn.ReLU(),
            # 第二个全连接层，将维度从seq_len // 10恢复到seq_len
            nn.Linear(seq_len // 10, seq_len),
        )

        self.fc_qry = nn.Sequential(
            # 第一个全连接层，将输入维度从seq_len降为seq_len // 10
            nn.Linear(seq_len, seq_len // 10),
            # 使用ReLU激活函数
            nn.ReLU(),
            # 第二个全连接层，将维度从seq_len // 10恢复到seq_len
            nn.Linear(seq_len // 10, seq_len),
        )
        # 定义特征融合的全连接层序列
        self.fc_fusion = nn.Sequential(
            # 第一个全连接层，将输入维度从seq_len * 2降为seq_len // 5
            nn.Linear(seq_len * 2, seq_len // 5),
            # 使用ReLU激活函数
            nn.ReLU(),
            # 第二个全连接层，将维度从seq_len // 5提升到seq_len
            nn.Linear(seq_len // 5, seq_len),
        )
        # 定义Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 计算两个特征之间的相关矩阵
    def correlation_matrix(self, spt_fg_fts, qry_fg_fts):
        spt_fg_fts = F.normalize(spt_fg_fts, p=2, dim=1)
        qry_fg_fts = F.normalize(qry_fg_fts, p=2, dim=1)
        # 计算支持集和查询集特征的逐元素相乘并在维度1上求和，得到余弦相似度矩阵
        cosine_similarity = torch.sum(spt_fg_fts * qry_fg_fts, dim=1, keepdim=True)
        return cosine_similarity

    def forward(self, spt_fg_fts, qry_fg_fts, band='None'):
        # 特征1通过全连接层并使用ReLU激活函数
        spt_proj = F.relu(self.fc_spt(spt_fg_fts))
        # 特征2通过全连接层并使用ReLU激活函数
        qry_proj = F.relu(self.fc_qry(qry_fg_fts))
        # 计算相关矩阵并通过Sigmoid函数，得到相似度矩阵
        similarity_matrix = self.sigmoid(self.correlation_matrix(spt_fg_fts, qry_fg_fts))

        if band == 'inhibit' :
            """
                【抑制】
                由于模型过度依赖这些频带中的突出特征会导致过拟合，在不同域间迁移时性能下降，
                    所以采用反向注意力加权（即使用 1 减去注意力矩阵后再与特征逐元素相乘），抑制相似部分，降低模型对这些域特定特征的依赖。
            """
            weighted_spt = (1 - similarity_matrix) * spt_proj
            weighted_qry = (1 - similarity_matrix) * qry_proj
        else:
            """
                【加强】
                直接将注意力矩阵与特征逐元素相乘，能够突出特征间的相似部分，增强模型对稳定、通用特征的捕捉。
            """
            # 特征1进行加权，权重为相似度矩阵
            weighted_spt = similarity_matrix * spt_proj
            # 特征2进行加权，权重为相似度矩阵
            weighted_qry = similarity_matrix * qry_proj

        # 将加权后的特征1和特征2在维度2上拼接
        combined = torch.cat((weighted_spt, weighted_qry), dim=2)
        # 对拼接后的特征通过融合全连接层并使用ReLU激活函数，得到融合后的张量
        fused_tensor = F.relu(self.fc_fusion(combined))

        return fused_tensor

if __name__ == '__main__':
    # 定义批量大小
    batch_size = 1
    # 定义特征维度
    feature_dim = 256
    # 定义元素数量
    H = 100
    W = 100
    num_elements = H * W

    # 创建AttentionMacthcing类的实例
    block = AttentionMacthcing(seq_len = num_elements)

    # 调用实例的前向传播函数，进行特征融合
    input1 = torch.rand(batch_size, feature_dim, H, W)
    print("input1.shape:", input1.shape)
    input2 = torch.rand(batch_size, feature_dim, H, W)
    print("input2.shape:", input2.shape)

    input1 = torch.reshape(input1, (batch_size, feature_dim, num_elements))
    input2 = torch.reshape(input2, (batch_size, feature_dim, num_elements))
    output = block(input1, input2, band='inhibit')
    output = torch.reshape(output, (batch_size, feature_dim, H, W))
    print("output.shape:", output.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
