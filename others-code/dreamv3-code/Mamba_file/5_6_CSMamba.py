import time
import math
import copy
from functools import partial
from typing import Optional, Callable

import timm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x)) # 执行平均池化:(B,d,H,W)-avg_pool->(B,d,1,1)-fc->(B,d,1,1)
        max_out = self.fc(self.max_pool(x)) # 执行最大池化:(B,d,H,W)-max_pool->(B,d,1,1)-fc->(B,d,1,1)
        out = avg_out + max_out # (B,d,1,1) + (B,d,1,1) == (B,d,1,1)
        return self.sigmoid(out) # 通过sigmoid生成权重表示:(B,d,1,1)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)   # 计算通道方向平均值:(B,d,H,W)-avg->(B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # 计算通道方向最大值:(B,d,H,W)-max->(B,1,H,W)
        x = torch.cat([avg_out, max_out], dim=1) # 通道方向拼接: (B,1,H,W)-cat-(B,1,H,W)-->(B,2,H,W);
        x = self.conv1(x) # 降维: (B,2,H,W)-->(B,1,H,W)
        return self.sigmoid(x) # 通过sigmoid生成权重表示:(B,1,H,W)


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=0.5,
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

        self.channel_attention = ChannelAttentionModule(self.d_inner)
        self.spatial_attention = SpatialAttentionModule()

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.forward_core = self.forward_corev0
        #self.forward_core = self.forward_core_windows
        # self.forward_core = self.forward_corev0_seq
        # self.forward_core = self.forward_corev1

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
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


    def forward_core_windows(self, x: torch.Tensor, layer=1):
        return self.forward_corev0(x)
        if layer == 1:
            return self.forward_corev0(x)
        downsampled_4 = F.avg_pool2d(x, kernel_size=2, stride=2)
        processed_4 = self.forward_corev0(downsampled_4)
        processed_4 = processed_4.permute(0, 3, 1, 2)
        restored_4 = F.interpolate(processed_4, scale_factor=2, mode='nearest')
        restored_4 = restored_4.permute(0, 2, 3, 1)
        if layer == 2:
            output = (self.forward_corev0(x) + restored_4) / 2.0

        downsampled_8 = F.avg_pool2d(x, kernel_size=4, stride=4)
        processed_8 = self.forward_corev0(downsampled_8)
        processed_8 = processed_8.permute(0, 3, 1, 2)
        restored_8 = F.interpolate(processed_8, scale_factor=4, mode='nearest')
        restored_8 = restored_8.permute(0, 2, 3, 1)

        output = (self.forward_corev0(x) + restored_4 + restored_8) / 3.0
        return output
        # B C H W

        num_splits = 2 ** layer
        split_size = x.shape[2] // num_splits  # Assuming H == W and is divisible by 2**layer

        # Use unfold to create windows
        x_unfolded = x.unfold(2, split_size, split_size).unfold(3, split_size, split_size)
        x_unfolded = x_unfolded.contiguous().view(-1, x.size(1), split_size, split_size)

        # Process all splits at once
        processed_splits = self.forward_corev0(x_unfolded)
        processed_splits = processed_splits.permute(0, 3, 1, 2)
        # Reshape to get the splits back into their original positions and then permute to align dimensions
        processed_splits = processed_splits.view(x.size(0), num_splits, num_splits, x.size(1), split_size, split_size)
        processed_splits = processed_splits.permute(0, 3, 1, 4, 2, 5).contiguous()
        processed_splits = processed_splits.view(x.size(0), x.size(1), x.size(2), x.size(3))
        processed_splits = processed_splits.permute(0, 2, 3, 1)

        return processed_splits


    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W

        K = 4
        ### 扫描展开 ###
        # 第一项: 按照行的顺序; 第二项: 按照列的顺序.
        # 第一项x1: (B,d,H,W)-view->(B,d,L);  第二项x2:(B,d,H,W)-transpose->(B,d,W,H)-view->(B,d,L);  最后将x1和x2在第一个维度进行stack:(B,2d,L)-view->(B,2,d,L);  相当于包含了两个不同排列方向的patch序列
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # x_hwwh翻转之后, 第一项:按照行的顺序逆排列; 第二项: 按照列的顺序逆排列。 以这样的方式, 拼接后就具有了四个方向的序列表示
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (B,2,d,L)-cat-(B,2,d,L) == (B,K,d,L);  K=4;   flip函数用于翻转某一维度

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) # (B,K,d,L)-einsum-(K,C,d) == (B,K,C,L)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) #将x_dbl分割为dts、B、C, 即(B,K,C,L)-split-> dts:(B,K,dt_rank,L); Bs:(B,K,d_state,L); Cs:(B,K,d_state,L)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # dts进行变换:(B,K,dt_rank,L)-einsum-(K,d,dt_rank) == (B,K,d,L)

        xs = xs.float().view(B, -1, L)  # (B,K,d,L)-view->(B,Kd,L)
        dts = dts.contiguous().float().view(B, -1, L)  # (B,K,d,L)-view->(B,Kd,L)
        Bs = Bs.float().view(B, K, -1, L)  # (B,K,d_state,L)-view->(B,K,d_state,L)
        Cs = Cs.float().view(B, K, -1, L)  # (B,K,d_state,L)-view->(B,K,d_state,L)

        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state) # (k*d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        ### S6 Block ###
        # xs包含了四个方向的patch排列
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L) # out_y:(B,K,d,L)
        assert out_y.dtype == torch.float

        ### 扫描合并 ###
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) # out_y[:, 2:4]: 将out_y的后两个序列(行的逆序,列的逆序)进行翻转,恢复正的顺序; (B,2,d,L), [行的正序、列的正序]
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) # 将按照列正顺序排列的序列进行翻转,恢复为默认的序列,即行优先; out_y[:, 1]: (B,d,L);  (B,d,L)-view->(B,d,W,H)-transpose->(B,d,H,W)-view->(B,d,L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) # 将inv_y中的第二个序列,也就是按照列正顺序排列的序列,恢复为默认的序列,即行优先;   inv_y[:, 1]: (B,d,L)-view->(B,d,W,H)-transpose->(B,d,H,W)-view->(B,d,L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y # out_y的第一个序列本就是按照行的正序排列的, inv_y[:, 0]从行的逆序纠正为行的正序, wh_y将列的正序恢复为行的正序, invwh_y将列的正序恢复为行的正序
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B,d,L)-transpose->(B,L,d)-view->(B,H,W,d)
        y = self.out_norm(y).to(x.dtype) # (B,H,W,d)

        return y

    def forward_corev0_seq(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)

        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.view(B, K, -1, L) # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        Ds = self.Ds.view(-1) # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1) # (k * d)

        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, layer=1, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x) # (B,H,W,C)-->(B,H,W,D)
        x, z = xz.chunk(2, dim=-1) # (B,H,W,D)--split-->x:(B,H,W,d) and z:(B,H,W,d);  D=2d,d=dinner

        z = z.permute(0, 3, 1, 2) # (B,H,W,d)-->(B,d,H,W)

        z = self.channel_attention(z) * z # 执行通道注意力: (B,d,H,W)-channel_attention->(B,d,1,1);  (B,d,1,1) * (B,d,H,W)==(B,d,H,W)
        z = self.spatial_attention(z) * z # 执行空间注意力: (B,d,H,W)-channel_attention->(B,1,H,W);  (B,1,H,W) * (B,d,H,W)==(B,d,H,W)
        z = z.permute(0, 2, 3, 1).contiguous() # 保存一下当前z值,它最后还要与SSM的输出相乘: (B,d,H,W)-->(B,H,W,d)

        x = x.permute(0, 3, 1, 2).contiguous() # (B,H,W,d)-->(B,d,H,W)
        x = self.act(self.conv2d(x)) # (B,d,H,W)-conv->(B,d,H,W)

        y = self.forward_core(x) # (B,d,H,W)-forward_core->(B,H,W,d)

        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            layer: int = 1,
            **kwargs,
    ):
        super().__init__()
        factor = 2.0
        d_model = int(hidden_dim // factor)
        self.down = nn.Linear(hidden_dim, d_model)
        self.up = nn.Linear(d_model, hidden_dim)
        self.ln_1 = norm_layer(d_model)
        self.self_attention = SS2D(d_model=d_model, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.layer = layer

    def forward(self, input: torch.Tensor):
        input_x = self.down(input) # (B,H,W,C)-->(B,H,W,d_model)
        input_x = input_x + self.drop_path(self.self_attention(self.ln_1(input_x))) #
        x = self.up(input_x) + input
        return x


if __name__ == '__main__':
    # (B,H,W,C)   B:batchsize;  C:通道数量
    x1 = torch.randn(1,224,224,64).to(device)

    Model = VSSBlock(
        hidden_dim=64,
        drop_path=0.1,
        norm_layer=nn.LayerNorm,
        attn_drop_rate=0.,
        d_state=16,
        # expand=0.25,
    ).cuda()

    out = Model(x1)
    print(out.shape)
