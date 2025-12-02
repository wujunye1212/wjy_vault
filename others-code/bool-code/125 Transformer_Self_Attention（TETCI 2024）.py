import torch
import torch.nn.functional as F
from torch import nn
"""
    论文地址：https://arxiv.org/pdf/2107.05274
    论文题目：TransAttUnet: Multi-level Attention-guided U-Net with Transformer for Medical Image Segmentation（TETCI 2024）
    中文题目：TransAttUnet：用于医学图像分割的具有变换器的多级注意力引导U-Net （TETCI 2024）
    讲解视频：https://www.bilibili.com/video/BV1vT9yYsEWS/
        Transformer 自注意力（Transformer Self Attention , TSA）：
            实际意义：①局部感知瓶颈：：卷积操作仅依赖局部邻域信息。
                    ②池化导致的信息丢失：下采样通过降低分辨率扩大感受野，但同时也丢失低层细节（边缘和纹理）。
            实现方式：将输入特征映射到多个子空间(多头)，每个头独立计算注意力，最后将结果拼接融合，允许模型并行捕捉不同子空间的语义关联。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""
class Transformer_Self_Attention(nn.Module):
    def __init__(self, temperature=512, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature ** 0.5  # 用于缩放注意力分数的温度参数
        self.dropout = nn.Dropout(attn_dropout)  # 注意力结果的随机失活

    def forward(self, x, mask=None):
        m_batchsize, d, height, width = x.size()  # 获取输入张量的批量大小、通道数、高度和宽度
        q = x.view(m_batchsize, d, -1)  # 将输入张量展开为查询向量

        k = x.view(m_batchsize, d, -1)  # 将输入张量展开为键向量
        k = k.permute(0, 2, 1)  # 转置键向量的最后两个维度

        v = x.view(m_batchsize, d, -1)  # 将输入张量展开为值向量
        attn = torch.matmul(q / self.temperature, k)  # 计算缩放后的注意力分数

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)  # 对需要忽略的区域设置负无穷大
        attn = self.dropout(F.softmax(attn, dim=-1))  # 对注意力分数进行归一化并应用Dropout

        output = torch.matmul(attn, v)  # 根据注意力分数加权值向量

        output = output.view(m_batchsize, d, height, width)  # 恢复原始的张量形状

        return output  # 返回自注意力模块的输出

if __name__ == '__main__':
    input = torch.rand(1, 64, 128, 128)  # 创建一个随机输入张量
    SAA = Transformer_Self_Attention()  # 初始化自注意力模块
    output = SAA(input)  # 计算自注意力模块的输出
    print(input.size())  # 打印输入张量的形状
    print(output.size())  # 打印输出张量的形状
    print("公众号、B站、CSDN同号")  # 输出推广信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 输出提示信息
