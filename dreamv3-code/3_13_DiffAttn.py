import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Union
import torch


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape # (B,2h,L,d)
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)



class MultiheadDiffAttn(nn.Module):
    def __init__(
            self,
            embed_dim,
            depth, # current layer index
            num_heads,
            num_kv_heads=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # arg num_heads set to half of baseline Transformer's num_heads
        # for e.g., to compare with a baseline Transformer with 16 heads, pass in num_heads=8 for DIFF Transformer
        self.num_heads = num_heads

        # arg num_kv_heads set to half of baseline Transformer's num_kv_heads if use GQA
        # for e.g., to compare with a baseline Transformer with 16 heads and 8 kv_heads,
        # pass in num_heads=8, num_kv_heads=4 for DIFF Transformer
        # if use MHA, pass in num_kv_heads=None
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0 ,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0 ,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0 ,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0 ,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, attn_mask=None):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len # 自注意力场景; 若扩展 cross-attn, 可以不同

        q = self.q_proj(x) # (B,L,C)-->(B,L,C)
        k = self.k_proj(x) # (B,L,C)-->(B,L,C)
        v = self.v_proj(x) # (B,L,C)-->(B,L,C)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)  # (B,L,C)-->(B,L,2h,d), C=2h*d
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim) # (B,L,C)-->(B,L,2h,d), C=2h*d
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim) # (B,L,C)-->(B,L,h,2*d), C=2h*d

        #q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        #k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.transpose(1, 2) # (B,L,2h,d)--transpose-->(B,2h,L,d)
        k = repeat_kv(k.transpose(1, 2), self.n_rep) # (B,L,2h,d)-transpose->(B,2h,L,d)--repeat_kv-->(B,2h,L,d)
        v = repeat_kv(v.transpose(1, 2), self.n_rep) # (B,L,h,2d)-transpose->(B,h,L,2d)--repeat_kv-->(B,h,L,2d)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) # (B,2h,L,d) @ (B,2h,d,L) == (B,2h,L,L)
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                    .float()
                    .fill_(float("-inf"))
                    .type_as(attn_weights),
                    1 + offset,
                    ) # 构造上三角自回归掩码 (严格因果)
        attn_weights = torch.nan_to_num(attn_weights) # (B,2h,L,L)
        attn_weights += attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        ) # 这里的 softmax 是对 2h 个头各自独立进行的，所以数学上等价于“两路分别 softmax”

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q) # 做点积得到单个标量，再分别指数
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q) # 做点积得到单个标量，再分别指数
        lambda_full = lambda_1 - lambda_2 + self.lambda_init # 相减加初始化
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len) # (B,2h,L,L)--view-->(B,h,2,L,L)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1] # 相减: (B,h,L,L)- λ·(B,h,L,L)

        attn = torch.matmul(attn_weights, v) # (B,h,L,L) @ (B,h,L,2d) == (B,h,L,2d)
        attn = self.subln(attn) # RMSNorm
        attn = attn * (1 - self.lambda_init) # 与论文一致的固定缩放，保持梯度规模对齐
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim) # (B,h,L,2d)-trans->(B,L,h,2d)-reshape->(B,L,2hd)==(B,L,C)

        attn = self.out_proj(attn) # (B,L,C)-->(B,L,C)
        return attn


if __name__ == '__main__':
    # (B,L,C)  B:batchsize  L:序列长度  C:通道的数量
    x1 = torch.randn(1,128,64)
    B,L,C = x1.size()


    Model = MultiheadDiffAttn(embed_dim=C, depth=1, num_heads=8)
    out= Model(x1) # out: (B,L,C)


    print(out.shape)