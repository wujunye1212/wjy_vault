# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
#from mamba_ssm_self.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    #from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from Mamba_file.mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
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

        ### 对应书中的 "Parameter"
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        # 这段代码通过生成一个随机dt值,并通过 softplus 的反函数计算得到对应的 inv_dt，然后将 inv_dt 赋值给 dt_proj.bias，以确保 dt_proj层的输出在经过 softplus 激活函数后输出在 dt_min 和 dt_max 之间。这对于模型的稳定性和性能至关重要,特别是在序列建模中，合理初始化这些参数可以帮助模型更快地收敛，并在训练初期避免数值不稳定
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor) # 将生成的dt值的最小值限制在 dt_init_floor 以上,以避免数值过小。

        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759, Add a numerically-stable function that computes the inverse of softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt)) #计算softplus的反函数值inv_dt，这个值将用于初始化 dt_proj.bias
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True # 标记dt_proj.bias不要在后续初始化过程中被重新初始化
        ###


        # S4D real initialization
        # 初始化状态空间模型（SSM）的参数 A, 以确保其在训练开始时处于合理的数值范围内,并且在数值计算过程中保持稳定,同时能够更好地捕捉到数据中的长程依赖关系。
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous() # (d_inner,d_state), 参数量是d_state个,然后复制了d_inner次
        A_log = torch.log(A)  # 计算A矩阵的对数，并将结果存储在 A_log中。这一步使得参数在数值上更加稳定，并有助于后续计算中保持精度。 在ssm中,A影响状态矩阵的变化,直接操作其指数形式可能导致数值不稳定，通过对数形式处理能让梯度的变化更平滑，更容易训练。
        self.A_log = nn.Parameter(A_log) # 将A_log转换为一个可训练的参数
        self.A_log._no_weight_decay = True # 确保在训练过程中不对其应用权重衰减（weight decay）

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape # (B,L,dim)

        conv_state, ssm_state = None, None
        # 推理阶段的时候执行
        if inference_params is not None:
            # #从缓存中获取当前层的卷积状态 (conv_state) 和状态空间模型状态 (ssm_state)
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            # 如果偏移量大于0,模型不再从头开始处理整个序列，而是只处理当前时间步的输入,此时需要逐步更新状态并生成当前时间步的输出, 在这种情况下只需要执行step函数即可
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time, (D,)
        # 将输入通过映射层得到xz:(D,dim) @ (dim,BL) == (D,BL)-rearrange->(B,D,L)   D= 2*d_inner
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # 将计算得到的A值取负数。这可能是为了在状态空间模型中引入一个稳定的衰减因子,确保状态不会无限增长。
        # 在初始化中, 先计算A矩阵对数,并将其作为可训练参数,确保数值稳定性,使梯度平滑易于训练。然后到这里重新调用计算好的A_log,取负值是为了确保状态空间模型中的状态演化具有衰减特性,使得状态不会无限制地增长,从而保持系统的稳定性。
        A = -torch.exp(self.A_log.float()) # (d_inner,d_state) 通过A_log重新计算矩阵 A,并且将结果取负值: (d_inner, d_state)

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        # 在 use_fast_path==True、causal_conv1d_fn导入的情况下、模型不处于推理模式的情况下,执行mamba_inner_fn
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            # 优化的核函数, 用于高效处理输入数据的卷积和状态更新。它集成了多个操作，并以一种紧凑的方式执行，以减少内存访问和计算开销。
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            ) # (B,D,L)-->(B,L,dim)
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
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
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    # S6计算  对新的token进行更新
    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B,d_inner)-->(B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1) # dt:(B,dt_rank)、B:(B,d_state)、C:(B,d_state)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B,dt_rank) @ (dt_rank,d_inner) = (B,d_inner)
        A = -torch.exp(self.A_log.float())  # 通过A_log重新计算矩阵A,并且将结果取负值: (d_inner, d_state), 取负值是为了确保状态空间模型中的状态演化具有衰减特性,使得状态不会无限制地增长,从而保持系统的稳定性。

        # SSM step
        if selective_state_update is None:
            # Discretize A and B, 与论文中的"Algorithm 2 SSM + Selection (S6)"进行对应: d_inner==D; d_state==N
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype)) # (B,d_inner)
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A)) # A矩阵的离散化: (B,d_inner)--einsum--(d_inner,d_state) == (B,d_inner,d_state)  这个过程不太好理解,大家可以GPT或者kimi问一下,给出的例子还是很清晰的
            dB = torch.einsum("bd,bn->bdn", dt, B) # B矩阵的离散化: (B,d_inner,d_state)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB) # h(t+1) = A * h(t) + B * x(t);    h(t+1):(B,d_inner,d_state)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C) # (B,d_inner,d_state)--einsum--(B,d_state) == (B,d_inner)
            y = y + self.D.to(dtype) * x  # skip连接  Y = C * h_t + D * x_t
            y = y * self.act(z)  # (B,D)  D==d_inner
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
