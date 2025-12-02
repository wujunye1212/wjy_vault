import torch
import torch.nn as nn
import einops
'''
    论文地址：https://arxiv.org/pdf/2303.15446.pdf
    论文题目：SwiftFormer: Efﬁcient Additive Attention for Transformer-based Real-time Mobile Vision Applications（ICCV 2023）
    中文题目：SwiftFormer：基于Transformer的实时移动视觉应用中的高效加性注意
    讲解视频：https://www.bilibili.com/video/BV1HVSEYDEzG/
         加性注意力机制:
         有效地用线性元素乘法替换了二次方的矩阵乘法运算。我们的设计表明，可以替换为一个线性层而不会牺牲任何准确性。
'''
class EfficientAdditiveAttnetion(nn.Module):
    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        # 线性变换层，将输入维度映射到token维度乘以头数
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        # 可学习的权重矩阵，用于计算query的重要性
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        # 缩放因子，通常为根号下的token维度的倒数
        self.scale_factor = token_dim ** -0.5
        # 投影层，保持维度不变
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        # 最终线性变换层，将结果压缩回原始token维度
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    # 前向传播函数
    def forward(self, x):

        #  torch.Size([3, 4096, 64])
        # 通过线性变换层得到 Q K
        Q = self.to_query(x)  # torch.Size([3, 4096, 64])
        K = self.to_key(x)      # torch.Size([3, 4096, 64])
        # 对Q、K进行归一化处理
        Q = torch.nn.functional.normalize(Q, dim=-1)  # 归一化最后一维 torch.Size([3, 4096, 64])
        K = torch.nn.functional.normalize(K, dim=-1)  # 归一化最后一维 torch.Size([3, 4096, 64])

        # ----------------------------------------------------------------------------
        # 图中左侧蓝色Q的全部计算
        # 计算查询与可学习权重w_g的点积，得到每个查询的重要性
        """
            Q        : torch.Size([3, 4096, 64])
            self.w_g : torch.Size([64, 1])
        """
        Q_weight = Q @ self.w_g  # BxNx1 (BxNxD @ Dx1)  torch.Size([3, 4096, 1])
        # 应用缩放因子，并且在行向量上归一化
        A = Q_weight * self.scale_factor  # BxNx1  torch.Size([3, 4096, 1]) * float
        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1    torch.Size([3, 4096, 1])
        # 计算全局描述符G，它是A和query的逐元素相乘后的总和
        G = torch.sum(A * Q, dim=1)  # BxD  torch.Size([3, 4096, 1]) * torch.Size([3, 4096, 64]) 求和
        # 重复G张量，使其形状匹配key，以便于后续运算
        G = einops.repeat(
            G, "b d -> b repeat d", repeat=K.shape[1]
        )  # BxNxD      torch.Size([3, 4096, 64])
        # ----------------------------------------------------------------------------

        # 图中右侧 相乘
        # 将G与key相乘后通过投影层，并加上原始查询
        """
            G :torch.Size([3, 4096, 64])
            K :torch.Size([3, 4096, 64])
            Q :torch.Size([3, 4096, 64])
        """
        out = self.Proj(G * K) + Q  # BxNxD torch.Size([3, 4096, 64])

        # 通过最终线性层输出
        out = self.final(out)  # BxNxD torch.Size([3, 4096, 32])
        return out


# 输入 B N C ,  输出 B N C
if __name__ == '__main__':
    block = EfficientAdditiveAttnetion(64, 32).cuda()  # 创建模型实例并移动到CUDA设备

    input = torch.rand(3, 64 * 64, 64).cuda()  # 创建随机输入张量并移动到CUDA设备
    output = block(input)  # 执行前向传播得到输出

    print(input.size())  # 打印输入张量尺寸
    print(output.size())  # 打印输出张量尺寸

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")