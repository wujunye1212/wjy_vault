import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import time
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Tuple


class RoPE(nn.Module):

    def __init__(self, embed_dim, num_heads):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4)) # 构造旋转角度向量
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    def forward(self, slen: Tuple[int]):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        # index = torch.arange(slen[0]*slen[1]).to(self.angle)
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        # sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
        # sin = sin.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :])  # (h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :])  # (w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1)  # (h w d1)
        # cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        # cos = cos.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :])  # (h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :])  # (w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1)  # (h w d1)

        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))

        return retention_rel_pos


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class AddLinearAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = self.head_dim ** -0.5
        self.elu = nn.ELU()


    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        '''
        x: (b c h w)
        sin: ((h w) d1)
        cos: ((h w) d1)
        '''
        B, C, H, W = x.shape
        qkvo = self.qkvo(x) # 使用一个 1×1 卷积生成 Q, K, V, O 四个部分的组合张量: (B,C,H,W)-->(B,4C,H,W)
        qkv = qkvo[:, :3*self.dim, :, :] # Q, K, V: (B,3C,H,W)
        o = qkvo[:, 3*self.dim:, :, :] # O: (B,C,H,W)
        lepe = self.lepe(qkv[:, 2*self.dim:, :, :]) # 通过 5x5 深度卷积为 V 添加局部空间位置信息: (B,C,H,W)-->(B,C,H,W)

        # 分离Q/K/V,并分头: (B,3C,H,W)-->(3,B,n,HW,d);  q = k = v = (B,n,HW,d)
        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads)

        q = self.elu(q) + 1  # 保证非负值,避免传统核函数如 exp(·) 的数值不稳定
        k = self.elu(k) + 1  # 强调注意力的强度信息（即幅值），避免被忽略

        # (B,n,HW,d)--mean-->(B,n,1,d)--transpose-->(B,n,d,1)
        # 计算幅值校正项, 用于改善注意力分布偏差: (B,n,HW,d) @ (B,n,d,1) == (B,n,HW,1), 论文中指出这一步能有效纠正幅值忽略的问题
        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale

        q = theta_shift(q, sin, cos) # 将二维相对位置编码 (sin, cos) 注入到 Q/K, 增强模型对空间位置关系的感知能力
        k = theta_shift(k, sin, cos) # (B,n,HW,d)

        # 核函数投影计算: (B,n,d,HW) @  (B,n,HW,d) = (B,n,d,d)
        kv = (k.transpose(-2, -1) * (self.scale / (H*W)) ** 0.5) @ (v * (self.scale / (H*W)) ** 0.5)

        # 残差校正与注意力输出: q @ kv = (B,n,HW,d) @ (B,n,d,d) = (B,n,HW,d)
        res = q @ kv * (1 + 1/(z + 1e-6)) - z * v.mean(dim=2, keepdim=True)

        # (B,n,HW,d)--rearrange-->(B,C,H,W)
        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        res = res + lepe # 加入 LePE 的局部位置信息
        return self.proj(res * o)




if __name__ == '__main__':
    # (B,C,H,W)
    x1 = torch.randn(1, 64, 224, 224)
    _, _, h, w = x1.size()

    # 定义 RoPE and AddLinearAttention
    RoPE = RoPE(embed_dim=64, num_heads=8)
    Model = AddLinearAttention(dim=64, num_heads=8)

    # 执行 RoPE and AddLinearAttention
    sin, cos = RoPE((h, w)) # 生成二维位置编码, sin:(HW, d), cos:(HW, d), d是每个注意力头的维度
    out = Model(x1, sin, cos) #
    print(out.shape)