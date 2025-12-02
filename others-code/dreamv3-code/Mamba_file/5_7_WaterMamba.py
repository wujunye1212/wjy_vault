import numbers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



## CCOSS
class ChannelMamba(nn.Module):
    def __init__(
        self,
        d_model,
        dim=None,
        d_state=16,
        d_conv=4,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        bimamba_type="v2",
        if_devide_out=False
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.if_devide_out = if_devide_out
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.bimamba_type = bimamba_type
        self.act = nn.SiLU()
        self.ln = nn.LayerNorm(normalized_shape=dim)
        self.ln1 = nn.LayerNorm(normalized_shape=dim)
        self.conv2d = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            bias=conv_bias,
            kernel_size=3,
            groups=dim,
            padding=1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        if bimamba_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

    def forward(self, u):
        """
        u: (B, H, W, C)
        Returns: same shape as hidden_states
        """
        b, d, h, w = u.shape # 将H看作通道, W和C看作特征图的高和宽
        l = h * w  # 把实际的W*C=C(W为1)看作序列长度,相当于把通道看作序列长度
        u = rearrange(u, "b d h w-> b (h w) d").contiguous()  # (B,H,1,C)-->(B,C,H)

        conv_state, ssm_state = None, None

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(u, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=l,
        ) # self.in_proj.weight:(2H,H)    u:(B,C,H)-rearrange->(H,BC)    矩阵乘法:(2H,H) @ (H,BC) = (2H,BC)   (2H,BC)-rearrange->(B,2H,C)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat

        x, z = xz.chunk(2, dim=1)  # (B,2H,C)--chunk-->x:(B,H,C) and z:(B,H,C)
        x = rearrange(self.conv2d(rearrange(x, "b l d -> b d 1 l")), "b d 1 l -> b l d") # (B,H,C)-rearrange->(B,C,1,H); (B,C,1,H)-conv->(B,C,1,H); (B,C,1,H)-rearrange->(B,H,C)

        # 生成两组(dt,B,C),类似两个全连接层的作用,映射到不同的向量空间
        # 第一组(dt,B,C)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  #  (B,H,C)-rearrange->(BC,H)-xproj->(BC,dt_rank+2d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1) # dt:(BC,dt_rank); B:(BC,d_state); C:(BC,d_state);
        dt = self.dt_proj.weight @ dt.t() # (H,dt_rank) @ (dt_rank,BC) == (H,BC)
        dt = rearrange(dt, "d (b l) -> b d l", l=l) # (H,BC)-->(B,H,C)
        B = rearrange(B, "(b l) d -> b d l", l=l).contiguous() # (BC,d_state)-->(B,d_state,C)
        C = rearrange(C, "(b l) d -> b d l", l=l).contiguous() # (BC,d_state)-->(B,d_state,C)

        # 第二组(dt_b,B_b,C_b)
        x_dbl_b = self.x_proj_b(rearrange(x, "b d l -> (b l) d"))  # (B,H,C)-rearrange->(BC,H)-x_proj_b->(BC,dt_rank+2d_state)
        dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1) # dt_b:(BC,dt_rank); B_b:(BC,d_state); C_b:(BC,d_state);
        dt_b = self.dt_proj_b.weight @ dt_b.t() # (H,dt_rank) @ (dt_rank,BC) == (H,BC)
        dt_b = rearrange(dt_b, "d (b l) -> b d l", l=l) # (H,BC)-->(B,H,C)
        B_b = rearrange(B_b, "(b l) d -> b d l", l=l).contiguous() # (BC,d_state)-->(B,d_state,C)
        C_b = rearrange(C_b, "(b l) d -> b d l", l=l).contiguous() # (BC,d_state)-->(B,d_state,C)

        # v1: 单向Mamba;  v2: 双向Mamba
        if self.bimamba_type == "v1":
            A_b = -torch.exp(self.A_b_log.float())
            out = selective_scan_fn(
                x,
                dt,
                A_b,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
        elif self.bimamba_type == "v2":
            A_b = -torch.exp(self.A_b_log.float())
            out = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ) # (B,H,C)-->(B,H,C)
            out_b = selective_scan_fn(
                x.flip([-1]),
                dt_b,
                A_b,
                B_b,
                C_b,
                self.D_b.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ) # (B,H,C)-->(B,H,C)
            out = self.ln(out) * self.act(z) # (B,H,C)-->(B,H,C)
            out_b = self.ln1(out_b) * self.act(z)  # (B,H,C)-->(B,H,C)
            if not self.if_devide_out:
                out = rearrange(out + out_b.flip([-1]), "b l (h w) -> b l h w", h=h, w=w) # 将反向序列重新翻转回来,然后变换shape: (B,H,C)-rearrange->(B,H,W,C)
            else:
                out = rearrange(out + out_b.flip([-1]), "b l (h w) -> b l h w", h=h, w=w) / 2

        return out


# channel coordinate omnidirectional selective scan
class CCOSS(nn.Module):
    def __init__(self,channel=3, w=128, h=128, d_state=16, expand=1, d_conv=4, mam_block=2):
        super().__init__()

        self.H_CSSM = nn.Sequential(*[
            ChannelMamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=h,
                dim=channel,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width
                expand=expand,  # Block expansion factor
            ) for i in
            range(mam_block)])

        self.W_CSSM = nn.Sequential(*[
            ChannelMamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=h,
                dim=channel,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width
                expand=expand,  # Block expansion factor
            ) for i in
            range(mam_block)])

        self.channel = channel
        self.ln = nn.LayerNorm(normalized_shape=channel)
        self.softmax = nn.Softmax(1)

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1,
                                  bias=False)

        self.dwconv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, groups=channel,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel)

        self.silu_h = nn.SiLU()
        self.silu_w = nn.SiLU()

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):

        x_s = x.contiguous()
        b, c, w, h = x.shape
        x = rearrange(x, "b c h w-> b (h w) c ") # 转换shape以便执行正则化: (B,C,H,W)-->(B,HW,C)
        x = (self.ln(x)) # 正则化
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w) # 正则化执行完毕后转换回来: (B,HW,C)-->(B,C,H,W)
        x_in = x # (B,C,H,W)
        x_shotcut = self.softmax(self.dwconv(x)) # (B,C,H,W)-->(B,C,H,W)
        x_h = torch.mean(x_in, dim=3, keepdim=True).permute(0, 1, 3, 2) # 在W方向上计算平均值: (B,C,H,W)-mean->(B,C,H,1)-permute->(B,C,1,H)
        x_w = torch.mean(x_in, dim=2, keepdim=True) # 在H方向上计算平均值:(B,C,H,W)-mean->(B,C,1,W)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3)))) # 拼接之后,执行conv-bn-relu: (B,C,1,H+W)
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3) # 重新分割为两部分: (B,C,1,H),(B,C,1,W)
        x_h = x_cat_conv_split_h.permute(0, 3, 2, 1) # (B,C,1,H)-permute->(B,H,1,C)
        x_h= self.H_CSSM(x_h).permute(0, 3, 2, 1) # 馈入到HCSSM模块: (B,H,1,C)-HCSSM->(B,C,1,H)-permute->(B,H,1,C); ChannelMamba的注释以H_CSSM为基础
        x_w = x_cat_conv_split_w.permute(0, 3, 2, 1) # (B,C,1,W)-permute->(B,W,1,C)
        x_w = self.W_CSSM(x_w).permute(0, 3, 2, 1) # 馈入到WCSSM模块: (B,W,1,C)-W_CSSM->(B,W,1,C)-permute->(B,C,1,W); ChannelMamba的注释以H_CSSM为基础,请在执行H_CSSM模的时候进去debug,而不是当前行执行debug
        s_h = self.sigmoid_h(x_h.permute(0, 1, 3, 2)) # 生成权重:(B,H,1,C)-permute->(B,H,C,1)-sigmoid_h->(B,H,C,1)
        s_w = self.sigmoid_w(x_w) # 生成权重:(B,C,1,W)-sigmoid_w->(B,C,1,W)
        out = s_h.expand_as(x) * s_w.expand_as(x) * x_shotcut # 调整输入:(B,C,H,W)*(B,C,H,W)*(B,C,H,W)==(B,C,H,W)

        return out + x_s


if __name__ == '__main__':
    # (B,C,H,W)   B:batchsize;  C:通道数量
    x1 = torch.randn(1, 64,224, 224).to(device)

    Model = CCOSS(channel=64, h=224, w=224, d_state=16, d_conv=4, mam_block=1).cuda()

    out = Model(x1)
    print(out.shape)

