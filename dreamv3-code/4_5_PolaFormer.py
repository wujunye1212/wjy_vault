import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops import rearrange


class PolaLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.format(sr_ratio, alpha, kernel_size))

    def forward(self, x, H, W):
        B, N, C = x.shape
        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2) # (B,N,C)-->(B,N,2*C); q:(B,N,C), g:(B,N,C)

        # 对 k/v 进行 spatial Reduction
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W) # (B,N,C)-permute->(B,C,N)-reshape->(B,C,H,W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) # (B,C,H,W)-sr->(B,C,H',W')-reshape->(B,C,N')-permute->(B,N',C)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3) # (B,N',C)-kv->(B,N',2*C)-reshape->(B,N',2,C)-permute->(2,B,N',C)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3) # (B,N,C)-kv->(B,N,2*C)-reshape->(B,N,2,C)-permute->(2,B,N,C)
        k, v = kv[0], kv[1] # k: (B,N,C);  v: (B,N,C);
        n = k.shape[1] # N, patch的个数

        k = k + self.positional_encoding.to(k.device) # 添加可学习的位置编码
        kernel_function = nn.ReLU() # ReLU 作为核函数 φ

        scale = nn.Softplus()(self.scale.to(x.device)) # 生成一个正数张量, 用于对 q 和 k 特征做放缩
        power = 1 + self.alpha * nn.functional.sigmoid(self.power) # power是一个可学习的参数,是用于控制极性注意力中正负特征的幅度增强程度的指数参数

        # 对 q/k 分别进行 scaling + 分头
        q = q / scale
        k = k / scale
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous() # (B,N,C)-reshape->(B,N,h,d)-permute->(B,h,N,d)
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous() # (B,N,C)-reshape->(B,N,h,d)-permute->(B,h,N,d)
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous() # (B,N,C)-reshape->(B,N,h,d)-permute->(B,h,N,d)

        # 分解极性分量（正负分开）, 放大或压缩 ReLU 激活后的特征值，增强注意力机制对重要特征的响应能力; 是解决传统线性注意力中 忽略负值导致信息缺失 的核心改进。
        q_pos = kernel_function(q) ** power # 正极性特征: (B,h,N,d)
        q_neg = kernel_function(-q) ** power # 负极性特征: (B,h,N,d)
        k_pos = kernel_function(k) ** power # 正极性特征: (B,h,N,d)
        k_neg = kernel_function(-k) ** power # 负极性特征: (B,h,N,d)

        # 构造极性注意力所需组合
        q_sim = torch.cat([q_pos, q_neg], dim=-1) # 对应同极性交互, (B,h,N,2d)
        q_opp = torch.cat([q_neg, q_pos], dim=-1) # 对应异极性交互, (B,h,N,2d)
        k = torch.cat([k_pos, k_neg], dim=-1) # 共用组合键, (B,h,N,2d)
        v1, v2 = torch.chunk(v, 2, dim=-1) # 将value矩阵划分为两部分,前者(B,h,N,d/2)用于处理同极性响应（正-正、负-负），后者(B,h,N,d/2)用于处理异极性响应（正-负、负-正）

        "同极性交互"
        # k矩阵变换: (B,h,N,2d)-mean->(B,h,1,2d)-transpose->(B,h,2d,1); q_sim与k矩阵进行运算:(B,h,N,2d) @ (B,h,2d,1) == (B,h,N,1);  这一步是事先计算分母,用作后续输出的缩放因子
        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        # k矩阵变换: (B,h,N,2d)-transpose->(B,h,2d,N); 计算加权 value 的映射矩阵（核函数映射）: (B,h,2d,N) @ (B,h,N,d/2) == (B,h,2d,d/2)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        # q_sim与 kv进行同极性运算, 然后除以分母: (B,h,N,2d) @ (B,h,2d,d/2) * (B,h,N,1) == (B,h,N,d/2)
        x_sim = q_sim @ kv * z

        "异极性交互"
        # k矩阵变换: (B,h,N,2d)-mean->(B,h,1,2d)-transpose->(B,h,2d,1); q_opp与k矩阵进行运算:(B,h,N,2d) @ (B,h,2d,1) == (B,h,N,1);  这一步是事先计算分母,用作后续输出的缩放因子
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        # k矩阵变换: (B,h,N,2d)-transpose->(B,h,2d,N); 计算加权 value 的映射矩阵（核函数映射）: (B,h,2d,N) @ (B,h,N,d/2) == (B,h,2d,d/2)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        # q_opp与 kv进行进行异极性运算, 然后除以分母: (B,h,N,2d) @ (B,h,2d,d/2) * (B,h,N,1) == (B,h,N,d/2)
        x_opp = q_opp @ kv * z

        # 拼接同极性和异极性的输出: (B,h,N,d/2)-cat-(B,h,N,d/2)==(B,h,N,d)
        x = torch.cat([x_sim, x_opp], dim=-1)
        # 恢复为与输入相同的shape: (B,h,N,d)-->(B,N,h,d)-reshape->(B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)

        # 如果 spatial Reduction 率大于1, 那么将其通过插值恢复原始shape:(B,N,C);
        if self.sr_ratio > 1:
            # (B,h,N,d)-tran->(B,h,d,N)-reshape->(Bh,d,n)-inter->(Bh,d,N)-reshape->(B,h,d,N)-trans->(B,h,N,d)
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N,
                                          mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)

        # (B,h,N,d)-reshape->(Bh,H,W,d)-permute->(Bh,d,H,W)
        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        # (Bh,d,H,W)-dwc->(Bh,d,H,W)-reshape->(B,C,N)-permute->(B,N,C)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        # 添加残差连接
        x = x + v
        # (B,N,C) * (B,N,C)
        x = x * g

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2) # (B,C,H,W)-proj->(B,D,H,W)-flatten->(B,D,HW)-transpose->(B,HW,D)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1] # 在这里重新更新了H和W的值: 14=224/16, 14=224/16

        return x, (H, W) # 输出patch化后的特征, 以及特征图的高和宽


if __name__ == '__main__':
    # (B,C,H,W)
    x1 = torch.randn(1, 64, 224, 224)

    # 定义一些中间参数
    in_channel = 64
    hidden_channel = 128
    # 定义PatchEmbed
    patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=in_channel, embed_dim=hidden_channel)
    # 定义PolaLinearAttention
    Model = PolaLinearAttention(hidden_channel, patch_embed.num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, kernel_size=5, alpha=4)

    # 执行PatchEmbed
    x, (H, W) = patch_embed(x1) # x:(B,N,D); N=H*W, 这里的H和W是被更新过的

    # 执行PolaLinearAttention
    out = Model(x, H, W) # (B,N,D)-->(B,N,C)

    # 查看输出的shape
    print(out.shape)