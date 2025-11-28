import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

'''
总结论文的主要贡献如下：
统一矩阵混合器视图：提出了一个统一的矩阵混合器视角，将序列混合层视为对输入序列的线性映射，
这一视角涵盖了从Transformer的自注意力到最新的结构化状态空间模型（SSMs）在内的多种序列模型。
矩阵参数化与序列对齐：引入了矩阵参数化的关键概念——序列对齐，这增强了模型的灵活性和性能，解释了Transformers和近期SSMs如Mamba的优异表现。
系统化设计方法：提供了基于矩阵混合器框架的系统化方法，用于开发具有特定属性的序列模型，促进了次二次复杂度模型的发展。
Hydra模型：提出了一种准分离矩阵混合器参数化的双向Mamba模型扩展，命名为Hydra，该模型在非因果任务上表现出色，超越了包括Transformers在内的其他序列模型。
实证结果：Hydra作为注意力层的替代品，在GLUE基准上比BERT提高了0.8点，在ImageNet数据集上的Top-1准确率比ViT提升了2%，展示了其在NLP和计算机视觉任务中的优越性能。
'''

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except ImportError:
    RMSNormGated = None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

class Hydra(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=7,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * (2 * self.ngroups * self.d_state) + 2 * self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * (2 * self.ngroups * self.d_state)
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        A = torch.ones(self.nheads, dtype=torch.float32, device=device)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        self.fc_D = nn.Linear(self.d_inner, self.nheads, bias=False, **factory_kwargs)

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=True, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        B, C,H,W = u.shape  #把四维张量，转化为三维张量

        u = u.reshape(B,C,H*W).transpose(-1,-2)
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        initial_states = repeat(self.init_states, "... -> b ...", b=2*batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * (2 * self.ngroups * self.d_state), 2 * self.nheads],
            dim=-1
        )

        dt = torch.cat((dt[:, :, :self.nheads], torch.flip(dt[:, :, self.nheads:], (1,))), dim=0)
        dt = F.softplus(dt + self.dt_bias)  # (2 * B, L, nheads)
        assert self.activation in ["silu", "swish"]

        # 1D Convolution
        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
        )  # (B, L, self.d_inner + 2 * (2 * ngroups * d_state))

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, BC = torch.split(xBC, [self.d_inner, 2 * (2 * self.ngroups * self.d_state)], dim=-1)
        x_og = x
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)
        BC = torch.cat(
            (BC[:, :, :2 * self.ngroups * self.d_state],
             torch.flip(BC[:, :, 2 * self.ngroups * self.d_state:], (1,))),
            dim=0
        )
        B, C = torch.split(BC, [self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=None,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        y = torch.roll(y, shifts=1, dims=1)
        y[:, 0, :] = 0.0
        y_fw, y_bw = y[:batch], torch.flip(y[batch:], (1,))
        y = y_fw + y_bw + x_og * repeat(
            F.linear(x_og, self.fc_D.weight, bias=self.D), "b l h -> b l (h p)", p=self.headdim
        )

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.out_proj(y)

        out = out.transpose(-1,-2).reshape(u.shape[0],u.shape[2],H,W)
        return out
if __name__ == '__main__':
    model = Hydra(64).cuda()
    inputs = torch.randn((1, 64, 32, 32)).cuda()
    pred = model(inputs)
    print(pred.size())
