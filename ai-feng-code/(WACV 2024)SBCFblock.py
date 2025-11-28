import torch
import torch.nn as nn
import itertools
from timm.models.layers import DropPath
# 代码：https://github.com/xyongLu/SBCFormer/blob/master/models/sbcformer.py
# 论文：https://arxiv.org/pdf/2311.03747
'''

                     WACV 2024顶会
计算机视觉在解决不同领域的现实问题方面越来越普遍，包括智能农业、渔业和畜牧业管理。
这些应用程序可能不需要每秒处理许多图像帧，因此从业者使用单板计算机 （SBC）。
尽管已经为移动/边缘设备开发了许多轻量级网络，但它们主要针对具有更强大处理器的智能手机。
本文介绍了一种名为 SBCFormer 的 CNN-ViT 混合网络，它可以在如此低端的 CPU 上实现高精度和快速计算。
这些 CPU 的硬件约束使 Transformer 的注意力机制比卷积更好。
然而，在低端 CPU 上使用注意力会带来挑战：高分辨率内部特征图需要过多的计算资源，但降低其分辨率会导致局部图像细节丢失。
SBCFormer 引入了一种体系结构设计来解决此问题。

SBCFormerBlock 的主要作用：
# 增强对局部和全局特征处理

局部特征处理作用：保留图像的局部细节，通过保持特征图的原始分辨率来实现。
全局特征处理作用：将特征图进行降采样，通过注意力机制捕捉全局信息，然后将特征图还原到原始尺寸。 
局部和全局特征在最后进行融合，生成既包含局部信息又包含全局信息的特征图，有助于提升模型的表现能力。
改进的注意力机制：SBCFormerBlock 还集成了一种修改过的注意力机制，设计目的是减少计算开销，
同时保持较好的效果。通过使用深度卷积来简化查询、键和值的操作，从而降低注意力计算的复杂度。

SBCFormerBlock 非常适合在资源受限（如低内存和低计算能力,移动端设备）的环境下的任务比如：实时检测，实时监控等。
适用于：目标检测、图像分类等所有CV任务，轻量高效的模块
'''
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class Conv2d_BN(nn.Module):
    def __init__(self, in_features, out_features=None, kernel_size=3, stride=1, padding=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

        # global FLOPS_COUNTER
        # output_points = ((resolution + 2 * padding - dilation *
        #                   (ks - 1) - 1) // stride + 1)**2
        # FLOPS_COUNTER += a * b * output_points * (ks**2) // groups

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 将 hidden_features 强制转换为整数
        self.fc1 = nn.Linear(in_features, int(hidden_features))
        self.act = act_layer()
        self.fc2 = nn.Linear(int(hidden_features), out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class InvertResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=3, act_layer=nn.GELU,
                 drop_path=0.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.pwconv1_bn = Conv2d_BN(self.in_features, self.hidden_features, kernel_size=1, stride=1, padding=0)
        self.dwconv_bn = Conv2d_BN(self.hidden_features, self.hidden_features, kernel_size=3, stride=1, padding=1,
                                   groups=self.hidden_features)
        self.pwconv2_bn = Conv2d_BN(self.hidden_features, self.in_features, kernel_size=1, stride=1, padding=0)

        self.act = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    # @line_profile
    def forward(self, x):
        x1 = self.pwconv1_bn(x)
        x1 = self.act(x1)
        x1 = self.dwconv_bn(x1)
        x1 = self.act(x1)
        x1 = self.pwconv2_bn(x1)

        return x + x1

class Attention(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8, attn_ratio=2, resolution=7):
        super().__init__()
        self.resolution = resolution
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.attn_ratio = attn_ratio
        self.scale = key_dim ** -0.5

        self.nh_kd = key_dim * num_heads
        self.qk_dim = 2 * self.nh_kd
        self.v_dim = int(attn_ratio * key_dim) * num_heads
        dim_h = self.v_dim + self.qk_dim

        self.N = resolution ** 2
        self.N2 = self.N
        self.pwconv = nn.Conv2d(dim, dim_h, kernel_size=1, stride=1, padding=0)
        self.dwconv = Conv2d_BN(self.v_dim, self.v_dim, kernel_size=3, stride=1, padding=1, groups=self.v_dim)
        self.proj_out = nn.Linear(self.v_dim, dim)
        self.act = nn.GELU()

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        h, w = self.resolution, self.resolution
        # 正确的 reshape
        x = x.transpose(1, 2).reshape(B, C, h, w)

        x = self.pwconv(x)
        qk, v1 = x.split([self.qk_dim, self.v_dim], dim=1)
        qk = qk.reshape(B, 2, self.num_heads, self.key_dim, N).permute(1, 0, 2, 4, 3)
        q, k = qk[0], qk[1]

        v1 = v1 + self.act(self.dwconv(v1))
        v = v1.reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.v_dim)
        x = self.proj_out(x)
        return x

class ModifiedTransformer(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, resolution=7):
        super().__init__()
        self.resolution = resolution
        self.dim = dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn = Attention(dim=self.dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                              resolution=resolution)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.mlp = Mlp(in_features=self.dim, hidden_features=self.dim * mlp_ratio, out_features=self.dim,
                       act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        # B, N, C = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SBCFormerBlock(nn.Module):  # building block
    def __init__(self, dim, resolution=7,depth_invres=2, depth_mattn=1, depth_mixer=2, key_dim=16, num_heads=3, mlp_ratio=4., attn_ratio=2,
                 drop=0., attn_drop=0.,drop_paths=[0.2], pool_ratio=1, invres_ratio=1,):
        super().__init__()
        self.resolution = resolution
        self.dim = dim
        self.depth_invres = depth_invres
        self.depth_mattn = depth_mattn
        self.depth_mixer = depth_mixer
        self.act = h_sigmoid()

        self.invres_blocks = nn.Sequential()
        for k in range(self.depth_invres):
            self.invres_blocks.add_module("InvRes_{0}".format(k),
                                          InvertResidualBlock(in_features=dim, hidden_features=int(dim * invres_ratio),
                                                              out_features=dim, kernel_size=3, drop_path=0.))

        self.pool_ratio = pool_ratio
        if self.pool_ratio > 1:
            self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
            self.convTrans = nn.ConvTranspose2d(dim, dim, kernel_size=pool_ratio, stride=pool_ratio, groups=dim)
            self.norm = nn.BatchNorm2d(dim)
        else:
            self.pool = nn.Identity()
            self.convTrans = nn.Identity()
            self.norm = nn.Identity()

        self.mixer = nn.Sequential()
        for k in range(self.depth_mixer):
            self.mixer.add_module("Mixer_{0}".format(k),
                                  InvertResidualBlock(in_features=dim, hidden_features=dim * 2, out_features=dim,
                                                      kernel_size=3, drop_path=0.))

        self.trans_blocks = nn.Sequential()
        for k in range(self.depth_mattn):
            self.trans_blocks.add_module("MAttn_{0}".format(k),
                                         ModifiedTransformer(dim=dim, key_dim=key_dim, num_heads=num_heads,
                                                             mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                                                             drop=drop, attn_drop=attn_drop, drop_path=drop_paths[k],
                                                             resolution=resolution))

        self.proj = Conv2d_BN(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_fuse = Conv2d_BN(self.dim * 2, self.dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, C, _, _ = x.shape
        h, w = self.resolution, self.resolution
        x = self.invres_blocks(x)
        local_fea = x

        if self.pool_ratio > 1.:
            x = self.pool(x)

        x = self.mixer(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.trans_blocks(x)
        x = x.transpose(1, 2).reshape(B, C, h, w)

        if self.pool_ratio > 1:
            x = self.convTrans(x)
            x = self.norm(x)
        global_act = self.act(self.proj(x))
        x_ = local_fea * global_act
        x_cat = torch.cat((x, x_), dim=1)
        out = self.proj_fuse(x_cat)

        return out

if __name__ == "__main__":

    # 实例化 SBCFormerBlock
    SBCFBlock = SBCFormerBlock(dim=64,resolution=32) #注意： 输入特征图的分辨率resolution=H=W
    # 创建一个随机的输入张量，形状为 B C H W
    input = torch.randn(2, 64, 32, 32)

    # 将输入张量通过 SBCFormerBlock
    output = SBCFBlock(input)

    # 打印输出张量的形状
    print(f"输入形状: {input.shape}")
    print(f"输出形状: {output.shape}")
