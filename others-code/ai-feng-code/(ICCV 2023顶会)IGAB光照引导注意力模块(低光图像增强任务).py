import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
'''
IGAB（光照引导注意力模块）是Retinexformer架构中的一个核心组成部分。
它的作用是在图像增强过程中，通过利用光照信息来指导非局部区域之间的交互建模。
具体来说，IGAB的设计目的是为了更好地处理不同光照条件下的区域间相互作用，
并且能够恢复低光照图像中的噪声、伪影和颜色失真等问题。
IGAB模块的原理如下：
1.层归一化：IGAB包含两个层归一化，用于对输入特征进行标准化处理，确保数据分布更加稳定，有助于网络训练。
2.光照引导多头自注意力机制（IG-MSA）：这是IGAB的核心部分。它使用了由One-stage Retinex-based Framework (ORF) 
捕获到的光照表示来指引自注意力计算。这意味着在处理每个像素时，不仅仅考虑其周围的局部上下文，还考虑到光照信息，
从而更准确地捕捉到图像中不同光照条件下的长距离依赖关系。
3.前馈网络（FFN）：继IG-MSA之后，IGAB采用了标准的前馈网络来进一步处理经过光照引导注意机制后的特征图，以提高模型的表现力。
IGAB通过整合光照信息与自注意力机制，旨在有效地解决低光照条件下图像中存在的各种问题，
如过曝、噪声放大及色彩失真等，最终输出高质量的增强图像。这种设计不仅简化了传统基于Retinex理论方法中复杂的多阶段流程，
同时也克服了卷积神经网络在捕捉远距离依赖方面的局限性。

'''

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)
class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p
        return out

class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            heads=1,
            num_blocks=1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out
#
if __name__ == '__main__':
    # 初始化IGAB模块
    model = IGAB(dim=32)
    # 随机创建两个输入特征图
    x_in = torch.randn(1,32,64,64)
    illu_fea_trans = torch.randn(1,32,64,64)
    # 前向传播
    output = model(x_in, illu_fea_trans)
    print('input_size:',x_in.size())
    print('output_size:',output.size())
