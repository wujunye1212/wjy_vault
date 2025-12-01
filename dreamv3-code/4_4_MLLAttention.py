import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1] #
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))  # (B,H,W,C)->reshape->(B,H,W,C/2,2)-view_as_complex->(B,H,W,C/2);  view_as_complex用于将输入张量的最后一个维度(正好有两个通道值)分别表示复数的实部和虚部
        pe_x = torch.view_as_complex(self.rotations) * x  # self.rotations:(H,W,C/2,2);  (H,W,C/2,2)-view_as_complex->(H,W,C/2);  (H,W,C/2) * (B,H,W,C/2) = (B,H,W,C/2)
        return torch.view_as_real(pe_x).flatten(-2) # view_as_real: 将复数数据转换为实数; (B,H,W,C/2)--view_as_real-->(B,H,W,C/2,2)-flatten->(B,H,W,C)


class LinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape # (B,L,C), L=n: 序列长度
        h = int(n ** 0.5) # n=h*w; h==w
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3) # 升维到2C,然后分割为q和k: (B,N,C)->(B,N,2C)-reshape->(B,N,2,C)-perumute->(2,B,N,C)
        q, k, v = qk[0], qk[1], x  # q,k,v: (B,N,C)


        q = self.elu(q) + 1.0 # (B,N,C)
        k = self.elu(k) + 1.0 # (B,N,C)
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3) # 为q矩阵添加旋转位置编码: (B,N,C)-reshape->(B,H,W,C)-rope->(B,H,W,C)-reshape->(B,N,h,d)-permute->(B,h,N,d);  N=H*W; c=h*d
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3) # 为k矩阵添加旋转位置编码: (B,N,C)-reshape->(B,H,W,C)-rope->(B,H,W,C)-reshape->(B,N,h,d)-permute->(B,h,N,d);  N=H*W; c=h*d
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3) # (B,N,C)-reshape->(B,N,h,d)-permute->(B,h,N,d)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3) # (B,N,C)-reshape->(B,N,h,d)-permute->(B,h,N,d)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3) # (B,N,C)-reshape->(B,N,h,d)-permute->(B,h,N,d)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6) # 先对k在N维度上进行平均池化,然后在q和k之间执行矩阵乘法(相当于原论文公式3中的分子的计算): (B,h,N,d) @ (B,h,d,1) == (B,h,N,1)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5)) # 先放缩,再计算kv乘积(相当于原论文公式3中的S的计算): (B,h,d,N) @ (B,h,N,d) = (B,h,d,d)
        x = q_rope @ kv * z # q与kv进行相乘,然后除以分子(在这里乘以分子的倒数):(B,h,N,d) @ (B,h,d,d) * (B,h,N,1) = (B,h,N,d)

        x = x.transpose(1, 2).reshape(b, n, c)  # 将注意力的输入X进行reshape: (B,h,N,d)--transpose->(B,N,h,d)-reshape->(B,N,C)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2) # 对value进行reshape: (B,h,N,d)--transpose->(B,H,W,C)-permute->(B,C,H,W)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c) # 执行Conv提取空间特征, 然后将其作为残差连接: (B,C,H,W)-lepe->(B,C,H,W)-permute->(B,H,W,C)-reshape->(B,N,C)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'





class MLLABlock(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # 在2D特征图上执行3×3Conv:(B,L,C)-reshape->(B,H,W,C)-permute->(B,C,H,W)-cpe1->(B,C,H,W);  重新展平,并添加残差(B,C,H,W)-flatten->(B,C,HW)-permute->(B,HW,C)==(B,L,C)
        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x  # (B,L,C)

        ### MLLA的第一步: 正则化  ###
        x = self.norm1(x)

        ### MLLA的第二步: 执行MLLA Block  ###
        # MLLA Block的右分支: 先通过线性层,然后接一个SiLU激活函数: (B,L,C)-act_proj->(B,L,C)-act->(B,L,C)
        act_res = self.act(self.act_proj(x))
        # MLLA Block的左分支第一步: 线性层变换: (B,L,C)-in_proj->(B,L,C)->(B,H,W,C)
        x = self.in_proj(x).view(B, H, W, C)
        # MLLA Block的左分支第二步: 执行3×3Conv,然后接一个SiLU激活函数: (B,H,W,C)-permute->(B,C,H,W)-dwc->(B,C,H,W)-permute->(B,H,W,C)-view->(B,L,C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C) #
        # MLLA Block的左分支第三步: 执行线性注意力: (B,L,C)-->(B,L,C)
        x = self.attn(x)
        # MLLA Block的左分支第四步: 线性注意力的输出与另一个分支进行相乘,并通过线性层得到输出: (B,L,C) * (B,L,C) = (B,L,C)
        x = self.out_proj(x * act_res)

        ### MLLA的第三步: 为MLLA Block的输出添加残差连接  ###
        x = shortcut + self.drop_path(x)

        ### MLLA的第四步: 执行第二个3×3Conv; 这一步模型中没有画出  ###
        # 在2D特征图上执行3×3Conv:(B,L,C)-reshape->(B,H,W,C)-permute->(B,C,H,W)-cpe1->(B,C,H,W);  重新展平,并添加残差(B,C,H,W)-flatten->(B,C,HW)-permute->(B,HW,C)==(B,L,C)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        ### MLLA的第五步: 执行Norm-FFN, 并添加残差连接 ###
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"



if __name__ == '__main__':
    # (B,L,C)   B:batchsize; L:序列长度,L=H*W   C:通道数量
    x1 = torch.randn(1,196,64).to(device)
    B,L,C = x1.size()
    H = W = int(L ** 0.5) # H*W=L; H和W是基于2D特征图的高和宽

    Model = MLLABlock(dim=C, input_resolution=(H,W), num_heads=8).to(device) # input_resolution=(H,W);  H*W=L; H和W是基于2D特征图的高和宽

    out = Model(x1)
    print(out.shape)