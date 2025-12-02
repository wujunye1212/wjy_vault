import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange



def degree(B, dim):
    bacth, n_heads, N, anchor_num = B.shape # (B,h,N,A)
    degreeB = torch.sum(B, dim=-2).pow(-1).unsqueeze(-1) # (B,h,N,A)-sum->(B,h,A)-pow->(B,h,A)-unsqueeze->(B,h,A,1)
    degreeB[torch.isinf(degreeB)] = 0.0 # pow(-1)是取倒数的操作，可能会出现无穷大，所以需要将无穷大的位置置0
    degreeB = degreeB.repeat(1, 1, 1, dim) # 使其恢复四维张量：(B,h,A,1)--repeat-->(B,h,A,D)
    return degreeB


class AnchorAttention(nn.Module):
    def __init__(self, dim, anchors=5, heads = 8, dim_head = 64, dropout = 0., k_sparse=5):
        super(AnchorAttention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.k_sparse = k_sparse
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.x_to_emb = nn.Linear(dim, inner_dim * 2, bias=False)
        self.dist_anchor_B = nn.Linear(dim_head, anchors, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        emv_v = self.x_to_emb(x).chunk(2, dim=-1) # (B,N,C)--x_to_emb->(B,N,h*D*2); chunk: (B,N,h*D), (B,N,h*D)
        q_k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), emv_v) # q_k, v: (B,N,h*D)-->(B,h,N,D)
        anchor_B = self.dist_anchor_B(q_k) * self.scale # A: anchor的数量; 生成锚点: (B,h,N,D)--dist_anchor_B-->(B,h,N,A)
        anchor_attn = anchor_B.softmax(dim=-1) # 通过softmax得到锚点概率矩阵：(B,h,N,A)
        attn_degree = degree(anchor_attn, dim=q_k.shape[-1]) # 计算每个锚点的度：(B,h,N,A)--degree-->(B,h,A,D)
        anchor_attn_T = anchor_attn.transpose(3, 2) # (B,h,N,A)--trans-->(B,h,A,N)

        out = anchor_attn_T.matmul(v) # 把所有token信息聚合到A个锚点上: (B,h,A,N) @ (B,h,N,D) == (B,h,A,D)
        out = attn_degree.mul(out) # 对锚点特征进行校准：(B,h,A,D) * (B,h,A,D) == (B,h,A,D)
        out = anchor_attn.matmul(out) # 将锚点信息重新分发到各个token上：(B,h,N,A) @ (B,h,A,D) == (B,h,N,D)

        out = rearrange(out, 'b h n d -> b n (h d)') # (B,h,N,D)--rearrange-->(B,N,h*D)
        return self.to_out(out) # (B,N,h*D)-->(B,N,C)


if __name__ == '__main__':
    # (B,N,C)
    x1 = torch.randn(1, 196, 64)
    B, N, C = x1.size()

    # 定义 AnchorAttention
    Model = AnchorAttention(dim=64)

    # 执行 AnchorAttention
    out = Model(x1)
    print(out.shape)