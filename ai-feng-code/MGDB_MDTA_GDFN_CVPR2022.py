import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

'''
来自CVPR顶会论文      
两个即插即用模块：MDTA 和 GDFN
           将MDTA和GDFN两个模块结合：取名为 MGDB

由于卷积神经网络（CNNs）能够从大规模数据中学习可泛化的图像先验，这些模型已被广泛应用于图像恢复及相关任务中。
最近，另一类神经架构——Transformer，在自然语言处理和高级视觉任务上展现出了显著的性能提升。
虽然Transformer模型缓解了CNNs的缺点（例如，有限的感受野和对输入内容的不适应性），
但其计算复杂度随空间分辨率的增大而呈二次增长，因此，对于大多数涉及高分辨率图像的图像恢复任务来说，
Transformer模型的应用是不可行的。

在这项工作中，我们提出了一种高效的Transformer模型，通过对构建块（多头注意力和前馈网络）进行几项关键设计，
使其能够捕捉长距离的像素交互，同时仍然适用于大图像。我们的模型，在多个图像恢复任务上实现了最先进的性能，
包括图像去雨、单图像运动去模糊、散焦去模糊（单图像和双像素数据）以及图像去噪（高斯灰度/彩色去噪和真实图像去噪）。

MDTA模块的主要作用包括：
1.线性复杂度：通过将自注意力机制应用于特征维度而非空间维度，MDTA模块显著降低了计算复杂度，
         使其具有线性复杂度。这使得MDTA模块能够高效地处理高分辨率图像。
2.全局上下文建模：虽然MDTA模块在空间维度上不显式地建模像素对之间的交互，但它通过计算特征通道之间的协方差来生成注意力图，
         从而隐式地编码全局上下文信息。这使得模型能够在不牺牲全局感受野的情况下，高效地捕捉图像中的长距离依赖关系。
3.局部上下文混合：MDTA模块在计算注意力图之前，通过1x1卷积和深度可分离卷积对输入特征进行局部上下文混合。
         这有助于强调空间局部上下文，并将卷积操作的互补优势融入到模型中。

GDFN模块的主要作用包括：
1.受控特征转换：通过引入门控机制，GDFN模块能够控制哪些互补特征应该向前流动，
        并允许网络层次结构中的后续层专注于更精细的图像属性。这有助于生成高质量的输出图像。
2.局部内容混合：与MDTA模块类似，GDFN模块也包含深度可分离卷积，用于编码来自空间相邻像素位置的信息。
            这有助于学习局部图像结构，对于有效的图像恢复至关重要。
上述模块适用于：图像恢复，图像去模糊，图像去噪，目标检测，图像分割，图像分类等所有计算机视觉CV任务通用的即插即用模块
'''


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type= 'WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(GDFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class MDTA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.size()

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
class MGDB(nn.Module): #TransformerBlock
    def __init__(self, dim):
        super(MGDB, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MDTA(dim)
        self.norm2 = LayerNorm(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # # 生成随机输入张量
    input = torch.randn(1, 32, 64, 64)
    # 实例化模型对象
    MDTA_model = MDTA(dim=32)
    GDFN_model = GDFN(dim=32)
    MGDB_model = MGDB(dim=32)  # MGDB 是MDTA和GDFN模块的结合

    # 执行 MDTA 前向传播
    output = MDTA_model(input)
    print('MDTA_input_size:', input.size())
    print('MDTA_output_size:', output.size())

    # 执行 GDFN 前向传播
    output = GDFN_model(input)
    print('GDFN_input_size:', input.size())
    print('GDFN_output_size:', output.size())

    # 执行 MGDB 前向传播
    output = MGDB_model(input)
    print('MGDB_input_size:', input.size())
    print('MGDB_output_size:', output.size())