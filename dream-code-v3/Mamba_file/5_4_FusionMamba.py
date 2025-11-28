import math
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2D选择性扫描模块
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        ### 扫描展开 ###
        # 第一项: 按照行的顺序; 第二项: 按照列的顺序
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)  # 第一项x1: (B,d,H,W)-view->(B,d,L);  第二项x2:(B,d,H,W)-transpose->(B,d,W,H)-view->(B,d,L);  最后将x1和x2在第一个维度进行stack:(B,2d,L)-view->(B,2,d,L);  相当于包含了两个不同排列方向的patch序列
        # x_hwwh翻转之后, 第一项:按照行的顺序逆排列; 第二项: 按照列的顺序逆排列。 以这样的方式, 拼接后就具有了四个方向的序列表示
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (B,2,d,L)-cat-(B,2,d,L) == (B,K,d,L);  K=4;   flip函数用于翻转某一维度

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)  # (B,K,d,L)-einsum-(K,C,d) == (B,K,C,L)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) #将x_dbl分割为dts、B、C, 即(B,K,C,L)-split-> dts:(B,K,dt_rank,L); Bs:(B,K,d_state,L); Cs:(B,K,d_state,L)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) #dts进行变换:(B,K,dt_rank,L)-einsum-(K,d,dt_rank) == (B,K,d,L)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (B,K,d,L)-view->(B,Kd,L)
        dts = dts.contiguous().float().view(B, -1, L)  # # (B,K,d,L)-view->(B,Kd,L)
        Bs = Bs.float().view(B, K, -1, L)  # (B,K,d_state,L)-view->(B,K,d_state,L)
        Cs = Cs.float().view(B, K, -1, L)  # (B,K,d_state,L)-view->(B,K,d_state,L)
        Ds = self.Ds.float().view(-1)  # (k*d)  K=4, d=dinner
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k*d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k*d)

        ### S6 Block ###
        # xs包含了四个方向的patch排列
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        ### 扫描合并 ###
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) # out_y[:, 2:4]: 将out_y的后两个序列(行的逆序,列的逆序)进行翻转,恢复正的顺序; (B,2,d,L), [行的正序、列的正序]
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) # 将按照列正顺序排列的序列进行翻转,恢复为默认的序列,即行优先; out_y[:, 1]: (B,d,L);  (B,d,L)-view->(B,d,W,H)-transpose->(B,d,H,W)-view->(B,d,L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) # 将inv_y中的第二个序列,也就是按照列正顺序排列的序列,恢复为默认的序列,即行优先;   inv_y[:, 1]: (B,d,L)-view->(B,d,W,H)-transpose->(B,d,H,W)-view->(B,d,L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class LDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        # conv.weight.size() = [out_channels, in_channels, kernel_size, kernel_size]
        super(LDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)  # kernel_size: (3,3)  weight:(D,D,K,K),D=channels,K=kernel-size; 在这里默认 D== in_channels == out_channels

        self.center_mask = torch.tensor([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]]).cuda() # 只保留卷积核对应窗口的最中心的信息:(K,K)
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)  # 初始化一个和conv层参数矩阵相同大小的全1矩阵:(D,D,K,K)
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
                                           requires_grad=True)  # 初始化一个(D,D)的可学习的mask矩阵
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)  # 初始化一个可学习的参数
        # print(self.learnable_mask[:, :, None, None].shape)

    def forward(self, x):

        # (D,D,K,K)- theta * (D,D,1,1) * (K,K) * (D,D,1,1) = (D,D,K,K)
        # theta * (D,D,1,1) * (K,K) * (D,D,1,1):  theta*(D,D,1,1)用于生成0-1之间的权重, 然后再与(K,K)相乘, 是为了调整卷积核最中心位置的权重, 最后再和(D,D,1,1)相乘, 这还是在调整卷积核最中心位置的权重, 只不过这次的权重来源于conv层卷积核数值的相加
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               self.center_mask * self.conv.weight.sum(2).sum(2)[:, :, None, None]

        out_diff = F.conv2d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)
        return out_diff


class eca_layer(nn.Module):

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.avg_pool(x) # 生成通道描述符表示:(B,C,H,W)-->(B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) # 通过conv建模通道间的相关性: (B,C,1,1)-->(B,C,1,1)
        y = self.sigmoid(y) # 通过sigmoid函数生成权重表示
        out = x * y.expand_as(x) # 调整输入
        return out


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x  # (B,H,W,C)
        x = self.norm(x)
        x_global = x.mean([1, 2], keepdim=True) # 生成通道描述符表示: (B,H,W,C)-->(B,1,1,C)
        x_global = self.act_fn(self.global_reduce(x_global)) #学习通道相关性, 先降维:(B,1,1,C)-->(B,1,1,d)
        # x_local = self.act_fn(self.local_reduce(x))
        c_attn = self.channel_select(x_global) # 学习通道相关性, 后升维:(B,1,1,d)-->(B,1,1,C)
        c_attn = self.gate_fn(c_attn) # 通过sigmoid生成权重
        attn = c_attn
        out = ori_x * attn # 权重调整输入
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.conv_branch = LDC(hidden_dim, hidden_dim)
        self.self_attention_cross_channel = eca_layer(channel=hidden_dim)
        self.se = BiAttn(hidden_dim)

    def forward(self, input: torch.Tensor):

        # 右分支
        x_ssm = self.drop_path(self.self_attention(self.ln_1(input)))  # 执行ESSM: (B,H,W,C)-->(B,H,W,C)
        x_ = x_ssm.permute(0,3,1,2) # (B,H,W,C)-->(B,C,H,W)
        x_ = self.self_attention_cross_channel(x_)  # 执行ECA:(B,C,H,W)
        x_ = x_.permute(0, 2, 3, 1) # (B,C,H,W)-->(B,H,W,C)
        x = x_ssm + x_ #右分支的残差连接

        # 左分支
        x_conv = self.conv_branch(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # 在这里是把右分支的输出x输入到左分支的, 如果你要严格按照论文讲述的来做, 可以把x换成input: (B,H,W,C)-permute->(B,C,H,W)-conv->(B,C,H,W)-permute->(B,H,W,C)

        # 融合
        x = self.se(x_ssm) + self.se(x_conv) # 实际这也是一个通道注意力层, 为两个分支都调整各自的特征表示: (B,H,W,C)-->(B,H,W,C)

        # 添加残差连接
        x = input + self.drop_path(x) # (B,H,W,C)

        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # H方向上从第0行开始下采样, W方向上从第0行开始下采样,步长为2: (B,H/2,W/2,C)
        x1 = x[:, 1::2, 0::2, :]  # H方向上从第1行开始下采样, W方向上从第0行开始下采样,步长为2: (B,H/2,W/2,C)
        x2 = x[:, 0::2, 1::2, :]  # H方向上从第0行开始下采样, W方向上从第1行开始下采样,步长为2: (B,H/2,W/2,C)
        x3 = x[:, 1::2, 1::2, :]  # H方向上从第1行开始下采样, W方向上从第1行开始下采样,步长为2: (B,H/2,W/2,C)

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # 将四个不同的特征图进行拼接: (B,H/2,W/2,4C), 拼接之后, 每个像素点的特征包含了
        x = x.view(B, H // 2, W // 2, 4 * C)  # (B,H/2,W/2,4C)-->(B,H/2,W/2,4C)

        x = self.norm(x)
        x = self.reduction(x) # (B,H/2,W/2,4C)-->(B,H/2,W/2,2C)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x) # (B,H/2,W/2,2C)-->(B,H/2,W/2,4C)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale) # 上采样恢复shape: (B,H/2,W/2,2C)-->(B,H,W,C)
        x = self.norm(x)

        return x


if __name__ == '__main__':
    # (B,H,W,C)   B:batchsize; H/W:既可以是特征图的高和宽,也可以是基于patch特征图的高和宽  C:通道数量
    x1 = torch.randn(1,14,14,64).to(device)

    # 下采样
    DownSample = PatchMerging2D(dim=64).cuda()
    # 模型
    Model = VSSBlock(hidden_dim=2 * 64, drop=0.,
                                attn_drop_rate=0., drop_path=0.1,
                                norm_layer=nn.LayerNorm, d_state=16).cuda()
    # 上采样
    UpSample = PatchExpand2D(dim=2 * 64).cuda()

    x = DownSample(x1) # 下采样, 特征图变小, 通道变多: (B,H,W,C) --> (B,H/2,W/2,2C)
    x = Model(x) # 执行SS2D,输入输出shape保持一致: (B,H/2,W/2,2C) --> (B,H/2,W/2,2C)
    out = UpSample(x) # 上采样恢复shape: (B,H/2,W/2,2C)-->(B,H,W,C)
    print(out.shape)