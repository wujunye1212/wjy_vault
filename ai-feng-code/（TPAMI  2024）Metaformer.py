import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple
from timm.models.layers import DropPath


'''
TPAMI 2024 顶刊论文   带来5个有效涨点改进
MetaFormer 是 Transformer 的抽象架构，已被发现在实现有竞争力的性能方面发挥着重要作用。
在本文中，我们进一步探索了 MetaFormer 的能力，再次将我们的注意力从令牌混合器设计中转移出来：
我们在 MetaFormer 下引入了几个使用最基本或最常见的混合器的基线模型，并展示了它们令人满意的性能。
在MetaFormer架构中将我们的观察结果总结如下：
1.CAFormer（Convolution-Attention Former）：
原理：CAFormer结合了卷积和自注意力机制。在模型的前两阶段使用深度可分离卷积进行局部信息混合，
在后两阶段则使用经典的自注意力机制进行全局信息混合。这种设计能够兼顾局部特征提取和全局信息建模。
作用：通过在不同阶段采用不同的token mixer，CAFormer在大规模图像分类任务上表现优异，达到了85.5%的顶级准确率。
2.ConvFormer：
原理：ConvFormer完全依赖于卷积作为token mixer，特别是使用深度可分离卷积。
这是一种纯卷积神经网络架构，未使用自注意力机制。
作用：ConvFormer通过卷积实现了高效的特征提取，且在参数量和计算复杂度上
优于一些强大的卷积神经网络模型如ConvNeXt，在ImageNet-1K上实现了竞争性的表现。
3.IdentityFormer：
原理：IdentityFormer使用身份映射（identity mapping）作为token mixer，即输入不经过任何形式的混合直接传递到后续层。
这是最简单的MetaFormer实现，用来探测MetaFormer架构的性能下界。
作用：尽管没有进行任何token混合，IdentityFormer仍然在ImageNet-1K上达到了80.4%的准确率，
展示了MetaFormer架构本身的稳健性和有效性。
4.PoolingFormer：
原理：PoolingFormer使用简单的池化操作作为token mixer，池化是一种局部操作，用于下采样特征并减小特征维度，同时保留重要信息。
作用：通过使用池化操作，PoolingFormer在计算效率和性能之间达到了良好的平衡，优于许多传统的深度学习模型。
5.RandFormer（RandomMixing Former）：
原理：RandFormer在前两阶段使用身份映射，在后两阶段使用随机矩阵作为token mixer。
随机矩阵会对token进行随机混合，以较低的计算开销实现信息传递。
作用：RandFormer通过随机混合提升了IdentityFormer的性能，在ImageNet-1K上获得了81.4%的准确率，
验证了MetaFormer架构在使用任意token mixer时的普适性和有效性。
这些模块各自代表了不同复杂度和效率的token混合策略，展示了MetaFormer架构的灵活性和性能潜力。

适用于：目标检测，图像分割，图像分类，图像增强等所有CV2维任务
'''
class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale
class SquaredReLU(nn.Module):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """

    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """

    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RandomMixing(nn.Module):
    def __init__(self, num_tokens=196, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1),
            requires_grad=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x


class LayerNormGeneral(nn.Module):
    r""" General LayerNorm for different situations.

    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default.
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance.
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.

        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """

    def __init__(self, affine_shape=None, normalized_dim=(-1,), scale=True,
                 bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster,
    because it directly utilizes otpimized F.layer_norm
    """

    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, kernel_size=7, padding=3,
                 **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modfiled for [B, H, W, C] input
    """

    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """

    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU,
                 norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x
class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x

