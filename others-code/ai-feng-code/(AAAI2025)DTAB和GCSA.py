import torch.nn.functional as F
from einops import rearrange
import numbers
import torch
from torch import einsum
import torch.nn as nn
'''
来自AAAI 2025顶会
即插即用模块： DTAB和GCSA  (整合全局与局部特征信息)

盲点网络（BSN）是自监督图像去噪（SSID）中普遍采用的神经架构。然而，大多数现有的BSN都是基于卷积层构建的。
尽管变换器在许多图像修复任务中显示出克服卷积限制的潜力，但其注意力机制可能会违反盲点要求，从而限制了它们在BSN中的适用性。
为此，我们提出分析并重新设计通道和空间注意力以满足盲点要求。具体来说，在多尺度架构中，下采样会将空间特征混入通道维度，
导致通道自注意力可能泄露盲点信息。为缓解这一问题，我们将通道分为若干组，并分别执行通道注意力。
对于空间自注意力，我们在注意力矩阵上应用一个精心设计的掩码来限制和模拟膨胀卷积的感受野。
基于重新设计的通道和窗口注意力，我们构建了一个基于变换器的盲点网络（TBSN），它展示了强大的局部拟合和全局视角能力。
此外，我们引入了一种知识蒸馏策略，将TBSN提炼成更小的去噪器，以提高计算效率，同时保持性能。
在真实世界图像去噪数据集上的广泛实验表明，TBSN大大扩展了感受野，并相对于最新的SSID方法表现出有利的性能。

DTAB（Dilated Transformer Attention Block）模块是论文中提出的一种创新模块，
主要克服传统卷积神经网络在自监督图像去噪任务中的局限性，并满足盲点网络（BSN, Blind-spot Network）的要求。
DTAB模块通过重新设计通道和空间自注意力机制来实现这一点，从而既能够增强局部适应能力，又可以大幅度扩展感受野。

DTAB的作用包括：
1.防止信息泄露：为了防止通道自注意力机制可能带来的盲点信息泄露问题，特别是在多尺度架构中，
     DTAB将通道分成若干组，并对每组分别执行通道注意力。这确保了即使当通道数量大于空间分辨率时，
     也不会出现空间信息的泄露，避免了过拟合到噪声输入的问题。
2.模拟膨胀卷积的感受野：对于空间自注意力机制，DTAB应用了一个精心设计的掩码到注意力矩阵上，
     以限制并模仿膨胀卷积的感受野。这意味着每个像素只能关注位于偶数坐标的其他像素，从而有效地保持了盲点要求，
     同时实现了比普通卷积更大的感受野和更强的局部拟合能力。

DTAB模块通过引入膨胀窗口和通道注意力机制，显著增强了模型的感受野和局部信息聚合能力，相较于先前的BSN模型表现出更好的性能。
      此外，它还支持一种知识蒸馏策略，可以在保持性能的同时大幅减少推理时的计算成本。
适用于：图像增强，图像去噪，图像恢复，目标检测，图像分割，图像分类，超分图像任务，语义分割，遥感语义分割等所有CV任务通用即插即用模块
'''

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h


class FixedPosEmb(nn.Module):
    def __init__(self, window_size, overlap_window_size):
        super().__init__()
        self.window_size = window_size
        self.overlap_window_size = overlap_window_size

        attention_mask_table = torch.zeros((window_size + overlap_window_size - 1), (window_size + overlap_window_size - 1))
        attention_mask_table[0::2, :] = float('-inf')
        attention_mask_table[:, 0::2] = float('-inf')
        attention_mask_table = attention_mask_table.view((window_size + overlap_window_size - 1) * (window_size + overlap_window_size - 1))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten_1 = torch.flatten(coords, 1)  # 2, Wh*Ww
        coords_h = torch.arange(self.overlap_window_size)
        coords_w = torch.arange(self.overlap_window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten_2 = torch.flatten(coords, 1)

        relative_coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.overlap_window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.overlap_window_size - 1
        relative_coords[:, :, 0] *= self.window_size + self.overlap_window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.attention_mask = nn.Parameter(attention_mask_table[relative_position_index.view(-1)].view(
            1, self.window_size ** 2, self.overlap_window_size ** 2
        ), requires_grad=False)
    def forward(self):
        return self.attention_mask


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class PatchUnshuffle(nn.Module):
    def __init__(self, p=2, s=2):
        super().__init__()
        self.p = p
        self.s = s

    def forward(self, x):
        n, c, h, w = x.shape
        x = nn.functional.pixel_unshuffle(x, self.p)
        x = nn.functional.pixel_unshuffle(x, self.s)
        x = x.view(n, c, self.p * self.p, self.s * self.s, h//self.p//self.s, w//self.p//self.s).permute(0, 1, 3, 2, 4, 5)
        x = x.contiguous().view(n, c * (self.p**2) * (self.s**2), h//self.p//self.s, w//self.p//self.s)
        x = nn.functional.pixel_shuffle(x, self.p)
        return x

class PatchShuffle(nn.Module):
    def __init__(self, p=2, s=2):
        super().__init__()
        self.p = p
        self.s = s

    def forward(self, x):
        n, c, h, w = x.shape
        x = nn.functional.pixel_unshuffle(x, self.p)
        x = x.view(n, c//(self.s**2), (self.s**2), (self.p**2), h//self.p, w//self.p).permute(0, 1, 3, 2, 4, 5)
        x = x.contiguous().view(n, c * (self.p**2), h//self.p, w//self.p)
        x = nn.functional.pixel_shuffle(x, self.s)
        x = nn.functional.pixel_shuffle(x, self.p)
        return x


class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class DilatedMDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(DilatedMDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

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


class DilatedOCA(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads, dim_head, bias):
        super(DilatedOCA, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,
            rel_size = window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )
        self.fixed_pos_emb = FixedPosEmb(window_size, self.overlap_win_size)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks, vs))

        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        #split heads
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_spatial_heads), (qs, ks, vs))

        # attention
        qs = qs * self.scale
        spatial_attn = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn += self.fixed_pos_emb()
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = (spatial_attn @ vs)

        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_spatial_heads, h = h // self.window_size, w = w // self.window_size, p1 = self.window_size, p2 = self.window_size)

        # merge spatial and channel
        out = self.project_out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=3, stride=1, dilation=2, padding=2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=3, stride=1, dilation=2, padding=2, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x

class GCSA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(GCSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

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

class DTAB(nn.Module):
    def __init__(self, dim, window_size=8, overlap_ratio=0.5, num_channel_heads=2, num_spatial_heads=2, spatial_dim_head=16, ffn_expansion_factor=1, bias=False, LayerNorm_type='BiasFree'):
        super(DTAB, self).__init__()


        self.spatial_attn = DilatedOCA(dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias)
        self.channel_attn = DilatedMDTA(dim, num_channel_heads, bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        x = x + self.spatial_attn(self.norm3(x))
        x = x + self.spatial_ffn(self.norm4(x))
        return x

# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    DTAB_module =  DTAB(64)
    GCSA_module = GCSA(64)
    input_tensor = torch.randn(1, 64, 128, 128)
    output_tensor = DTAB_module(input_tensor)
    print('DTAB_Input size:', input_tensor.size())  # 打印输入张量的形状
    print('DTAB_Output size:', output_tensor.size())  # 打印输出张量的形状

    output_tensor = GCSA_module(input_tensor)
    print('GCSA_Input size:', input_tensor.size())  # 打印输入张量的形状
    print('GCSA_Output size:', output_tensor.size())  # 打印输出张量的形状