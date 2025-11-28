import math

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import _cfg, Mlp
from timm.models.registry import register_model
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

from einops import rearrange, repeat


class AttentionTSSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
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

        # (B,N,C)--qkv-->(B,N,C); (B,N,C)-->(B,N,h,d)-->(B,h,N,d);  C=h*d,h是注意力头个数,d是每个头的通道数
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        b, h, N, d = w.shape

        w_normed = torch.nn.functional.normalize(w, dim=-2) # (B,h,N,d)
        w_sq = w_normed ** 2  # 通过“平方”得到能量：(B,h,N,d)-->(B,h,N,d)

        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)  # 平方后的通道求和，然后每个注意力头使用不同的temp进行调整，最后在注意力头的方向上进行softmax归一化（得到的是每个token在不同注意力头上的概率）：(B,h,N,d)--sum-->(B,h,N)--attend-->(B,h,N)

        # 得到每个注意力头的总和：(B,h,N)--sum-->(B,h,1);
        # 归一化：[(B,h,N)/(B,h,1)]--unsqueeze-->(B,h,1,N);
        # 利用W^2做带权平均，得到每个头的“二阶矩统计量”：(B,h,1,N) @ (B,h,N,d) == (B,h,1,d)
        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        # 得到对角缩放矩阵
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        # 得到最终的输出：(B,h,N,d) * (B,h,N,1) * (B,h,1,d) = (B,h,N,d)
        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        out = rearrange(out, 'b h n d -> b n (h d)') # (B,h,N,d)-->(B,N,C)
        return self.to_out(out)


if __name__ == '__main__':
    # (B,N,C)
    x1 = torch.randn(1, 196, 64)
    B, N, C = x1.size()

    # 定义 AttentionTSSA
    Model = AttentionTSSA(dim=64)

    # 执行 AttentionTSSA
    out = Model(x1)
    print(out.shape)