# 论文题目：Multimodal Fusion Transformer for Remote Sensing Image Classification
# https://github.com/AnkurDeria/MFT?tab=readme-ov-file
# https://arxiv.org/pdf/2203.16952

import torch.nn as nn
import torch
from einops import rearrange

'''
题目：面向遥感图像分类的多模态融合变换器    IEEE 2023
多头交叉块注意力模块：MCPA
本文介绍了一种用于遥感图像（RS）数据融合的新型多头交叉补丁注意力（mCrossPA）机制。
类tokens还包含补充信息，源自多模态数据（例如 LiDAR、MSI、SAR 和 DSM），
这些数据与 HSI 补丁tokens信息一起馈送到vit网络。 新开发的 mCrossPA 
广泛使用了的注意力机制，可以有效地将来自 HSI 补丁tokens和现有 CLS tokens的信息
融合到一个集成了多模态特性的新tokens中。

在各种cv和nlp任务上是通用即插即用模块
'''

class MCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(head_dim, dim , bias=qkv_bias)
        self.wk = nn.Linear(head_dim, dim , bias=qkv_bias)
        self.wv = nn.Linear(head_dim, dim , bias=qkv_bias)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:N, ...].reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        x = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2)
        x = x.reshape(B, N, C * self.num_heads)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

if __name__ == '__main__':
    model = MCrossAttention(dim=32)
#   1.如果输入的是图片四维数据需转化为三维数据，演示代码如下
#     H, W = 32, 32
#     input = torch.randn(1, 32, H, W)
#     input_3d = to_3d(input)
#     output_3d = model(input_3d)
#     output_4d = to_4d(output_3d, H, W)
#     print('input_size:', input.size())
#     print('output_size:', output_4d.size())

#   2.如果输入的是三维数据演示代码如下
    input = torch.randn(1,1024,32) #B ,L,N
    output = model(input)
    print('input_size:',input.size())
    print('output_size:',output.size())

