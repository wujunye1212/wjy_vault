import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
layer_scale = True
init_value = 1e-6
'''
来自AAAI 2025顶会
即插即用注意力模块：SSA稀疏自注意力模块

利用非语义信息往往在局部和全局之间保持一致性，同时相较于语义信息在图像不同区域表现出更大的独立性，
SparseViT 提出了以稀疏自注意力为核心的架构，取代传统 Vision Transformer (ViT) 的全局自注意力机制，
通过稀疏计算模式，使得模型自适应提取图像篡改检测中的非语义特征。
Sparse Self-Attention(SSA)其主要作用如下：
1.适应性提取非语义特征：Sparse Self-Attention机制旨在从被操纵的图像中自适应地提取非语义特征（即与图像内容无关但对图像操纵敏感的特征）。
            这种机制不需要依赖手工设计的特征提取器，从而提高了模型在未知或复杂场景中的泛化能力。
2.减少计算负担：通过稀疏化注意力计算，Sparse Self-Attention减少了模型的浮点运算次数（FLOPs），降低了计算资源的需求，
           同时保留了高效的特征捕获能力，特别是在复杂场景中表现卓越。这使得模型更加高效，特别是在处理大规模图像数据时。
3.聚焦于局部不一致性：传统全局注意力机制倾向于关注图像的整体语义结构，在进行全局交互时可能会忽略掉由操纵引起的局部非语义信息的不一致性。
          SSA 通过限制注意力计算到特定的张量块内，抑制了语义信息的表达，使模型能够更专注于捕捉这些局部的、由操纵引入的非语义特征。
4.增强模型对操纵痕迹的敏感度：通过精准提取非语义信息，SSA增强了模型对图像操纵留下的细微痕迹的敏感度，
          从而显著提升了模型在不同数据集上的表现。
5.支持多尺度特征融合：Sparse Self-Attention允许在不同阶段使用不同的稀疏率，这有助于模型自适应地根据不同程度的语义信息抑制来提取各种非语义特征，
         增强了模型跨不同视觉场景的泛化能力，并且通过多尺度监督进一步提高了模型性能。
适用于所有计算机视觉CV任务，通用注意力模块！
'''

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print(x.shape)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def block(x, block_size):
    B, H, W, C = x.shape
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.reshape(B, Hp // block_size, block_size, Wp // block_size, block_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x, H, Hp, C

def unblock(x, Ho):
    B, H, W, win_H, win_W, C = x.shape
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H * win_H, W * win_W, C)
    Wp = Hp = H * win_H
    Wo = Ho
    if Hp > Ho or Wp > Wo:
        x = x[:, :Ho, :Wo, :].contiguous()
    return x

def alter_sparse(x, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    assert x.shape[1] % sparse_size == 0 & x.shape[2] % sparse_size == 0, 'image size should be divisible by block_size'
    grid_size = x.shape[1] // sparse_size
    out, H, Hp, C = block(x, grid_size)
    out = out.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = out.reshape(-1, sparse_size, sparse_size, C)
    out = out.permute(0, 3, 1, 2)
    return out, H, Hp, C

def alter_unsparse(x, H, Hp, C, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(-1, Hp // sparse_size, Hp // sparse_size, sparse_size, sparse_size, C)
    x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = unblock(x, H)
    out = out.permute(0, 3, 1, 2)
    return out

class SSA(nn.Module):
    def __init__(self, dim, num_heads=4, sparse_size=2, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        self.sparse_size = sparse_size
        if self.ls:
            global init_value
            # print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x_befor = x.flatten(2).transpose(1, 2)
        B, N, H, W = x.shape
        if self.ls:
            x, Ho, Hp, C = alter_sparse(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)
            x = x_befor + self.drop_path(self.gamma_1 * x)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))
        else:
            x, Ho, Hp, C = alter_sparse(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)
            x = x_befor + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x
# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    module = SSA(dim=64)
    input_tensor = torch.randn(1, 64, 128, 128)
    output_tensor = module(input_tensor)
    print('Input size:', input_tensor.size())  # 打印输入张量的形状
    print('Output size:', output_tensor.size())  # 打印输出张量的形状
