import torch
import torch.nn as nn
import torch.nn.functional as  F
import numbers
from einops import rearrange, repeat
import math


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except ImportError as e:
    pass
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import warnings
from ultralytics.nn.modules import C2f, C3

warnings.filterwarnings('ignore')
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

'''
来自CVPR2025顶会论文
即插即用模块：EVSS高效的视觉扫描模块

本文主要内容：  教大家使用5步法写顶会顶刊摘要，提高中稿率

尽管卷积神经网络（CNN）和视觉Transformer（ViT）在图像恢复任务中取得了优异的性能，
但Transformer因其能够捕捉长距离依赖关系和输入相关特性，通常在图像恢复任务中优于CNN。#第一步：交代任务，及目前现有的先进模型方法

然而，Transformer的计算复杂度随图像分辨率呈二次增长，限制了其在高分辨率图像恢复中的实际应用价值。#第二步：指出现有模型方法的不足

为此，本文提出了一种简单而高效的视觉状态空间模型（EVSSM）用于图像去模糊，旨在将状态空间模型（SSM）的优势引入视觉任务。
                                                                             #第三步：引出本文创新点，提出自己的XX任务的模型或框架
与现有采用多方向扫描以提取特征的方法不同，这类方法计算成本较高，本文设计了一种高效的视觉扫描模块（EVS），
该模块在每个SSM模块前应用不同的几何变换，从而在保持高效率的同时捕捉有用的非局部信息。      #第四步：使用1到2句话简单介绍本文创新点

大量实验结果表明，所提出的EVSSM在多个基准数据集和真实模糊图像上，相较现有先进图像去模糊方法表现更优。
                                                                          #第五步：通过广泛定性和定量实验表明，我们的方法具有较好的效果

EVS模块和EDFFN模块是EVSS中关键的组成部分，分别负责高效建模图像的非局部空间信息和增强频域特征表达。
EVS模块通过在每次状态扫描前对输入特征施加简单的几何变换（如翻转和转置），结合Mamba提出的Selective Scan机制，
仅通过单方向扫描即可实现多方向信息感知，显著降低计算成本同时提升非局部建模能力。

EDFFN模块则在前馈网络的末端引入频域筛选操作，将特征变换至频域后进行判别性筛选，
再通过逆变换还原至时域，有效保留图像的高频细节，提升去模糊效果，同时保持较高的运行效率。

EVSS、EVS和EDFFN模块适合：图像恢复、图像去雨、暗光增强、目标检测、图像分割、遥感语义分割等所有CV任务通用的即插即用模块
'''
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        b, c, h, w = x.shape
        h_n = (8 - h % 8) % 8
        w_n = (8 - w % 8) % 8

        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)

        x = x[:, :, :h, :w]

        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=8,
            d_conv=3,
            expand=2.,
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
        self.act = nn.GELU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.x_conv = nn.Conv1d(in_channels=(self.dt_rank + self.d_state * 2),
                                out_channels=(self.dt_rank + self.d_state * 2), kernel_size=7, padding=3,
                                groups=(self.dt_rank + self.d_state * 2))

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

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

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape, Bs.shape, Cs.shape, Ds.shape, dts.shape)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, 'b h w c -> b c h w')

        return out


##########################################################################

class EVSblock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False, LayerNorm_type='WithBias', att=True, idx=3, patch=128):
        super(EVSblock, self).__init__()

        self.att = att
        self.idx = idx
        if self.att:
            self.norm1 = LayerNorm(dim)
            self.attn = SS2D(d_model=dim, patch=patch)

        self.norm2 = LayerNorm(dim)
        self.ffn = EDFFN(dim, ffn_expansion_factor, bias)

        self.kernel_size = (patch, patch)

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        # assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def forward(self, x):
        if self.idx % 2 == 1:
            x = torch.flip(x, dims=(-2, -1)).contiguous()
        if self.idx % 2 == 0:
            x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()
        #            if self.idx % 4 == 3:
        #                x = torch.flip(x, dims=(-2, -1)).contiguous()
        #            if self.idx % 4 == 0:
        #                x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()

        x = self.grids(x)
        x = x + self.attn(self.norm1(x))
        x = self.grids_inverse(x)

        return x
class EVSS(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False, LayerNorm_type='WithBias', att=False, idx=3, patch=128):
        super(EVSS, self).__init__()

        self.att = att
        self.idx = idx
        if self.att:
            self.norm1 = LayerNorm(dim)
            self.attn = SS2D(d_model=dim, patch=patch)

        self.norm2 = LayerNorm(dim)
        self.ffn = EDFFN(dim, ffn_expansion_factor, bias)

        self.kernel_size = (patch, patch)

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        # assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def forward(self, x):
        if self.att:

            if self.idx % 2 == 1:
                x = torch.flip(x, dims=(-2, -1)).contiguous()
            if self.idx % 2 == 0:
                x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()
            #            if self.idx % 4 == 3:
            #                x = torch.flip(x, dims=(-2, -1)).contiguous()
            #            if self.idx % 4 == 0:
            #                x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()

            x = self.grids(x)
            x = x + self.attn(self.norm1(x))
            x = self.grids_inverse(x)

        x = x + self.ffn(self.norm2(x))

        return x
if __name__ == '__main__':
    block = EVSS(64,att=True).to('cuda')
    input = torch.rand(1, 64, 32, 32).to('cuda')
    output = block(input)
    print('EVSS input_size:',input.size())
    print('EVSS output_size:',output.size())

    block = EVSblock(64).to('cuda')
    input = torch.rand(1, 64, 32, 32).to('cuda')
    output = block(input)
    print('EVSblock input_size:',input.size())
    print('EVSblock output_size:',output.size())

    block = EDFFN(64).to('cuda')
    input = torch.rand(1, 64, 32, 32).to('cuda')
    output = block(input)
    print('EDFFN input_size:', input.size())
    print('EDFFN output_size:', output.size())