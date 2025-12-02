import math
import logging
from functools import partial
from typing import Optional
import torch
import torch.nn as nn



class CrossAttention(nn.Module):
    """Cross-Attention between latents and input tokens -- returning the refined latents and tokens as tuple """
    def __init__(self, dim_lat, dim_pat, dim_attn, num_heads=8, rv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim_attn % num_heads == 0, 'dim_attn MUST be divisible by num_heads'

        self.num_heads = num_heads
        head_dim = dim_attn // num_heads
        self.scale = head_dim ** -0.5
        self.dim_attn = dim_attn

        self.rv_latents = nn.Linear(dim_lat, dim_attn * 2, bias=rv_bias)  # 'in-projection' for latents
        self.rv_patches = nn.Linear(dim_pat, dim_attn * 2, bias=rv_bias)  # 'in-projection' for patches/tokens
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_dropT = nn.Dropout(attn_drop)
        self.proj_lat = nn.Linear(dim_attn, dim_lat)             # 'out-projection' for latents
        self.proj_drop_lat = nn.Dropout(proj_drop)
        self.proj_pat = nn.Linear(dim_attn, dim_pat)             # 'out-projection' for patches/tokens
        self.proj_drop_pat = nn.Dropout(proj_drop)

    def forward(self, x_latents, x_patches):
        B_lat, N_lat, _ = x_latents.shape  # Note: need B_lat since 1 at very first pass, then broadcasted/extended to bs
        B_pat, N_pat, _ = x_patches.shape
        rv_lat = self.rv_latents(x_latents).reshape(B_lat, N_lat, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4) # (B,N,C)-rv->(B,N,2C)-reshape->(B,N,2,h,d)-permute->(2,B,h,N,d)
        r_lat, v_lat = rv_lat.unbind(0) # (2,B,h,N,d)--unbind-->(B,h,N,d) and (B,h,N,d)
        rv_pat = self.rv_patches(x_patches).reshape(B_pat, N_pat, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4) # (B,N,C)-rv->(B,N,2C)-reshape->(B,N,2,h,d)-permute->(2,B,h,N,d)
        r_pat, v_pat = rv_pat.unbind(0) # (2,B,h,N,d)--unbind-->(B,h,N,d) and (B,h,N,d)

        # attention: (q@k.T), and will be multiplied with the value associated with the keys k
        attn = (r_lat @ r_pat.transpose(-2, -1)) * self.scale  # query from latent, key from patches: (B,h,N,d) @ (B,h,d,N) == (B,h,N,N)
        attn_T = attn.transpose(-2, -1)  # bidirectional attention, associated with the values from the query q: (B,h,N,N)--trans-->(B,h,N,N)

        attn = attn.softmax(dim=-1)  # softmax along patch token dimension
        attn_T = attn_T.softmax(dim=-1)  # softmax along latent token dimension

        attn = self.attn_drop(attn)
        attn_T = self.attn_dropT(attn_T)

        # Retrieve information form the patch tokens via latent query:
        x_latents = (attn @ v_pat).transpose(1, 2).reshape(-1, N_lat, self.dim_attn) # (B,h,N,N) @ (B,h,N,d) == (B,h,N,d); (B,h,N,d)-trans->(B,N,h,d)-reshape->(B,N,C)
        x_latents = self.proj_lat(x_latents) # (B,N,C)--proj-->(B,N,C)
        x_latents = self.proj_drop_lat(x_latents)

        # Likewise, store information from the latents in the patch tokens via transposed attention:
        x_patches = (attn_T @ v_lat).transpose(1, 2).reshape(B_pat, N_pat, self.dim_attn) # (B,h,N,N) @ (B,h,N,d) == (B,h,N,d); (B,h,N,d)-trans->(B,N,h,d)-reshape->(B,N,C)
        x_patches = self.proj_pat(x_patches) # (B,N,C)--proj-->(B,N,C)
        x_patches = self.proj_drop_pat(x_patches)

        return x_latents, x_patches


if __name__ == '__main__':
    # (B,N,C)
    x1 = torch.randn(1, 196, 64)
    x2 = torch.randn(1, 196, 64)
    B,N,C = x1.size()

    # 定义 CrossAttention
    Model = CrossAttention(dim_lat=C, dim_pat=C, dim_attn=C)

    # 执行 AttentionTSSA
    x_latents, x_patches = Model(x1, x2)
    print(x_latents.shape, x_patches.shape)