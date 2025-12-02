from typing import Tuple
import torch
import torch.nn as nn
from einops import rearrange

"""
    论文地址：https://arxiv.org/abs/2507.00698
    论文题目：Rectifying Magnitude Neglect in Linear Attention（ICCV 2025）
    中文题目：修正线性注意力中的幅度忽略问题（ICCV 2025）
    讲解视频：https://www.bilibili.com/video/BV15XpLzkE2a
    幅度感知线性注意力（Magnitude-Aware Linear Attention, MALA）：
        实际意义：①线性注意力忽视幅度问题：线性注意力（Linear Attention）在计算过程中完全忽略了查询（Query）的幅度信息，这导致线性注意力与传统的Softmax Attention在注意力得分分布上的显著差异，导致注意力分配不均，表现出较弱的局部感知能力。
                ②注意力得分分布不合理的问题：由于忽视查询（Query）幅度，不会随着幅度变化而发生动态调整，导致它不能和Softmax Attention一样有效捕捉到局部特征。
        实现方式：原文中出现大量数学推导，但是只是想使用它，因此以代码为准。
"""

def rotate_every_two(x):
    # 获取偶数列
    x1 = x[:, :, :, ::2]
    # 获取奇数列
    x2 = x[:, :, :, 1::2]
    # 将x1和-x2堆叠，形成旋转效果
    x = torch.stack([-x2, x1], dim=-1)
    # 将最后两维展平
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    # 计算基于sin和cos的旋转偏移
    return (x * cos) + (rotate_every_two(x) * sin)

class RoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # 计算位置编码角度
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    def forward(self, slen: Tuple[int]):
        # 创建索引h和w
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)

        # 计算sin函数
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :])  # (h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :])  # (w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1)  # (h w d1)

        # 计算cos函数
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :])  # (h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :])  # (w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1)  # (h w d1)

        # 返回计算好的sin和cos，便于后续使用
        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))
        return retention_rel_pos

class MALAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # 初始化模型的维度和头数
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 定义卷积层
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = self.head_dim ** -0.5  # 缩放因子
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        # 获取输入张量的形状
        B, C, H, W = x.shape

        # 计算qkvo，获取q, k, v
        qkvo = self.qkvo(x)  # (b 4*c h w)
        qkv = qkvo[:, :3 * self.dim, :, :]
        o = qkvo[:, 3 * self.dim:, :, :]  # b c h w

        # 通过卷积获取lepe
        lepe = self.lepe(qkv[:, 2 * self.dim:, :, :])  # (b c h w)

        # 重排qkv的维度
        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads)  # (b n (h w) d)
        # 使用ELU激活函数
        q = self.elu(q) + 1
        k = self.elu(k) + 1
        # 计算注意力得分z：用于在计算过程中进行平衡，确保注意力得分合理分配。
        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale

        # 对q和k进行旋转偏移
        # 不一样的地方
        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)
        # 计算kv
        kv = (k.transpose(-2, -1) * (self.scale / (H * W)) ** 0.5) @ (v * (self.scale / (H * W)) ** 0.5)
        # 计算最终的结果 公式13
        res = q @ kv * (1 + 1 / (z + 1e-6)) - z * v.mean(dim=2, keepdim=True)
        # 重排结果并加上lepe
        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)

        res = res + lepe

        return self.proj(res * o)

if __name__ == "__main__":
    # 创建一个随机输入张量
    x = torch.randn(1, 32, 50, 50)

    # 初始化RoPE和MALAttention
    RoPE = RoPE(embed_dim=32, num_heads=8)
    sin, cos = RoPE((50, 50))  # 生成sin和cos用于位置编码

    # 初始化MALAttention并进行前向传播
    MALA = MALAttention(dim=32, num_heads=8)
    output = MALA(x, sin, cos)

    # 打印输入和输出张量的形状
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")