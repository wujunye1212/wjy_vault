import time
import math
from functools import partial
from typing import Optional, Callable
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass


# https://github.com/YubiaoYue/MedMamba/blob/main/MedMamba.py

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                             dim=1).view(B, 2, -1, L) # 第一项x1: (B,d,H,W)-view->(B,d,L);  第二项x2:(B,d,H,W)-transpose->(B,d,W,H)-view->(B,d,L);  最后将x1和x2在第一个维度进行stack:(B,2d,L)-view->(B,2,d,L);  相当于包含了两个不同排列方向的patch序列
        # x_hwwh翻转之后, 第一项:按照行的顺序逆排列; 第二项: 按照列的顺序逆排列。 以这样的方式, 拼接后就具有了四个方向的序列表示
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B,2,d,L)-cat-(B,2,d,L) == (B,K,d,L);  K=4;   flip函数用于翻转某一维度

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) # (B,K,d,L)-einsum-(K,C,d) == (B,K,C,L)
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

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y  # out_y的第一个序列本就是按照行的正序排列的, inv_y[:, 0]从行的逆序纠正为行的正序, wh_y将列的正序恢复为行的正序, invwh_y将列的正序恢复为行的正序

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (B,K,d,L)-view->(B,Kd,L)
        dts = dts.contiguous().float().view(B, -1, L)  # # (B,K,d,L)-view->(B,Kd,L)
        Bs = Bs.float().view(B, K, -1, L)  # (B,K,d_state,L)-view->(B,K,d_state,L)
        Cs = Cs.float().view(B, K, -1, L)  # (B,K,d_state,L)-view->(B,K,d_state,L)
        Ds = self.Ds.float().view(-1)  # (k*d)  K=4, d=dinner
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k*d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)   # (k*d)


        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L) # out_y:(B,K,d,L)
        assert out_y.dtype == torch.float


        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape  # (B,H,W,C)

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B,d,L)-transpose->(B,L,d)-view->(B,H,W,d)
        y = self.out_norm(y) # (B,H,W,d)
        y = y * F.silu(z) # z通过silu激活函数, 用于调整y
        out = self.out_proj(y) # (B,H,W,d)-->(B,H,W,C)
        if self.dropout is not None:
            out = self.dropout(out)
        return out



def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, height, width, num_channels = x.size() # (B,H,W,C)
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, height, width, groups, channels_per_group) # (B,H,W,C)-->(B,H,W,2,C/2)

    x = torch.transpose(x, 3, 4).contiguous() # (B,H,W,2,C/2)-->(B,H,W,C/2,2)

    # flatten
    x = x.view(batch_size, height, width, -1) # (B,H,W,C)

    return x



class SS_Conv_SSM(nn.Module):
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
        self.ln_1 = norm_layer(hidden_dim//2)
        self.self_attention = SS2D(d_model=hidden_dim//2, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

        self.conv33conv33conv11 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim // 2),
            nn.Conv2d(in_channels=hidden_dim//2,out_channels=hidden_dim//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=1, stride=1),
            nn.ReLU()
        )
        # self.finalconv11 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)
    def forward(self, input: torch.Tensor):
        input_left, input_right = input.chunk(2,dim=-1) # (B,H,W,C)--> (B,H,W,C/2) and (B,H,W,C/2)
        x = self.drop_path(self.self_attention(self.ln_1(input_right))) # (B,H,W,C/2)
        input_left = input_left.permute(0,3,1,2).contiguous() # (B,H,W,C/2)-permute->(B,C/2,H,W)
        input_left = self.conv33conv33conv11(input_left) # (B,C/2,H,W)--conv33conv33conv11-->(B,C/2,H,W)
        input_left = input_left.permute(0,2,3,1).contiguous() # (B,C/2,H,W)--permute-->(B,H,W,C/2)
        output = torch.cat((input_left,x),dim=-1) # (B,H,W,C/2)--cat-- (B,H,W,C/2)--> (B,H,W,C)
        output = channel_shuffle(output,groups=2)
        return output+input

if __name__ == '__main__':
    # (B,H,W,C)
    x1 = torch.randn(1,224,224,64).to(device)

    Model = SS_Conv_SSM(hidden_dim=64).to(device)
    out = Model(x1)
    print(out.shape)