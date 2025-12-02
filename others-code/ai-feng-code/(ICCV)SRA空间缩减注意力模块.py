import torch
import torch.nn as nn
from timm.models.layers import DropPath
# https://arxiv.org/abs/2102.12122
# 代码：https://github.com/whai362/PVT/tree/v2

'''
即插即用模块：SRA空间缩减注意力模块   ICCV
与 MHA 类似，我们的 SRA 接收查询 Q、键 K 和值 V 作为输入，并输出优化后的特征。
区别在于，我们的 SRA 缩小了 K 和 V 的空间尺度，用于处理高分辨率特征图并降低计算及内存成本.
因此我们的SRA可以用有限的资源处理更大的输入特征图/序列.

适用于：纯transform的检测和分割，图像分类，语义分割，目标检测等所有CV2维任务
'''

class SRAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == '__main__':
    # 定义输入参数
    dim = 64  # 输入通道数
    num_heads = 8  # 注意力头数量
    H, W = 16, 16  # 输入特征图的高度和宽度
    sr_ratio = 2  # 降采样比例
    # 创建SRAttention模块
    sr_attention = SRAttention(dim, num_heads, sr_ratio=sr_ratio)
    # 创建输入张量
    input = torch.randn(1, H * W, dim)
    output = sr_attention(input, H, W)
    print('input_size:',input.size())
    print('output_size:',output.size())
