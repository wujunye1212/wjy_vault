import torch
from torch import nn
from einops import rearrange
"""
    论文地址：https://arxiv.org/abs/2412.17810
    论文题目：Token Statistics Transformer: Linear-Time Attention via Variational Rate Reduction（ICLR 2025）
    中文题目：令牌统计变换器：通过变分率降低实现线性时间注意力机制（ICLR 2025）
    讲解视频：https://www.bilibili.com/video/BV12DQcYEEEz/
        令牌统计自注意力（Token Statistics Self-Attention，TSSA）：
            实际意义：①计算复杂度高：传统 Transformer 的自注意力机制计算所有Token之间的相似性，计算量和内存需求会大幅增加，限制了模型的扩展和应用。
                    ②模型的可解释性：传统方法可解释性差。
            实现方式（优势）：1）TSSA 算子通过全新的设计，避免两两相似性的计算，其计算复杂度与 token 数量呈线性关系，在计算时间和内存使用上都显著优于传统注意力机制。
                            2）与传统注意力机制相比，TSSA 算子是通过数学公式推导形成可解释注意力模式，无需复杂的自监督训练。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""
class AttentionTSSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,  attn_drop=0., proj_drop=0.):
        """
        :param dim: 输入特征的维度
        :param num_heads: 注意力头的数量，默认为 8
        :param qkv_bias: 是否在 QKV 投影中使用偏置，默认为 False
        :param attn_drop: 注意力矩阵的 dropout 概率
        :param proj_drop: 输出投影的 dropout 概率
        """
        super().__init__()  # 调用父类的初始化方法

        self.heads = num_heads  # 保存注意力头的数量

        # 定义一个 Softmax，用于计算注意力权重
        self.attend = nn.Softmax(dim=1)
        # 定义一个 Dropout，用于对注意力权重进行随机丢弃
        self.attn_drop = nn.Dropout(attn_drop)

        # 定义一个线性层，用于生成 QKV 矩阵
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)

        # 定义一个可学习的参数 temp，用于调整注意力计算
        self.temp = nn.Parameter(torch.ones(num_heads, 1))

        # 定义输出投影，包括一个线性层和一个 Dropout
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),  # 线性层，用于将维度映射回原始输入维度
            nn.Dropout(proj_drop)  # Dropout，用于随机丢弃部分输出
        )

    def forward(self, x):
        # 在通道维度上分成多头
        # [batch_size, seq_length, dim] ===> [batch_size, heads, seq_length, head_dim]
        # torch.Size([1, 784, 64])      ===> torch.Size([1, 8, 784, 8])
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        # 对 w 沿着最后一个维度进行归一化，标准化操作
        # torch.Size([1, 8, 784, 8])
        w_normed = torch.nn.functional.normalize(w, dim=-2)
        # 对归一化后的 w 进行平方
        w_sq = w_normed ** 2
        # 计算注意力权重 Pi：对 w_sq 沿着最后一个维度求和后乘以 temp，再通过 Softmax
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)  # 形状为 [batch_size, heads, seq_length]

        # 论文中注意力算子的相关计算步骤
        # 该算子：实现低秩投影，避免计算令牌间成对相似性，具有线性计算和内存复杂度。
        # 计算注意力得分 dots: Pi 先进行归一化，再扩展一个维度，与 w 的平方相乘。
        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        # 计算注意力矩阵 attn，公式中为 1 / (1 + dots)
        attn = 1. / (1 + dots)
        # 对注意力矩阵进行 dropout 操作，防止过拟合。
        attn = self.attn_drop(attn)

        # 计算输出，公式中为 -w * Pi * attn
        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        # 将输出重新排列为 [batch_size, seq_length, dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 将输出通过输出投影层
        return self.to_out(out)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


if __name__ == '__main__':
    batch_size = 1
    channel = 64
    H = 28
    W = 28
    seq_length = H*W
    input = torch.rand(1, 64, 28, 28)
    print('input_size:', input.size())
    # 1 X 64 X 28 X 28 ====> 1 X 784 X 64
    input = to_3d(input)

    # 初始化 AttentionTSSA 模型
    model = AttentionTSSA(dim=channel)
    output = model(input)
    output = to_4d(output, H, W)
    print('output_size:', output.size())

    # 计算模型的总参数量，并打印
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')
    print("公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")