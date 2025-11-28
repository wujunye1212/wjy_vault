import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def num_patches(seq_len, patch_len, stride):
    return (seq_len - patch_len) // stride + 1


class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False, drop_out=0.7):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            if drop_out != 0:
                layer_list.append(nn.Dropout(drop_out))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


class DeformAtten1D(nn.Module):
    '''
        max_offset (int): The maximum magnitude of the offset residue. Default: 14.
    '''
    def __init__(self, seq_len, d_model, n_heads, dropout, kernel=5, n_groups=4, no_off=False, rpb=True) -> None:
        super().__init__()
        self.offset_range_factor = kernel
        self.no_off = no_off
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_groups = n_groups
        self.n_group_channels = self.d_model // self.n_groups
        self.n_heads = n_heads
        self.n_head_channels = self.d_model // self.n_heads
        self.n_group_heads = self.n_heads // self.n_groups
        self.scale = self.n_head_channels ** -0.5
        self.rpb = rpb

        self.proj_q = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Linear(self.d_model, self.d_model)
        kernel_size = kernel
        self.stride = 1
        pad_size = kernel_size // 2 if kernel_size != self.stride else 0
        self.proj_offset = nn.Sequential(
            nn.Conv1d(self.n_group_channels, self.n_group_channels, kernel_size=kernel_size, stride=self.stride, padding=pad_size),
            nn.Conv1d(self.n_group_channels, 1, kernel_size=1, stride=self.stride, padding=pad_size),
        )

        self.scale_factor = self.d_model ** -0.5  # 1/np.sqrt(dim)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.d_model, self.seq_len))
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        dtype, device = x.dtype, x.device
        x = x.permute(0 ,2 ,1) # (B,L,C)--permute-->(B,C,L)

        q = self.proj_q(x) # 生成q矩阵: (B,C,L)--proj_q-->(B,C,L)

        group = lambda t: rearrange(t, 'b (g d) n -> (b g) d n', g = self.n_groups)  # 将通道分组
        grouped_queries = group(q) # 将q矩阵进行分组,以提高计算效率: (B,C,L)--group-->(gB,d,L); C=g*d,g是组的数量,d是每组通道数

        offset = self.proj_offset(grouped_queries) # 通过卷积生成偏移量, 论文中的η_{off}, 输出的序列长度为N: (gB,d,L)--proj_offset-->(gB,1,L)
        offset = rearrange(offset, 'b 1 n -> b n') # offset表示每组的时间步偏移: (gB,1,L)-->(gB,L)

        # feats:(gB,d,L)
        def grid_sample_1d(feats, grid, *args, **kwargs):
            # does 1d grid sample by reshaping it to 2d
            grid = rearrange(grid, '... -> ... 1 1') # 扩展为2D: (gB,L)-->(gB,L,1,1), 为了匹配 F.grid_sample()所要求的 2D 输入格式
            grid = F.pad(grid, (1, 0), value = 0.) # (gB,L,1,1)-->(gB,L,1,2)
            feats = rearrange(feats, '... -> ... 1') # (gB,d,L)-->(gB,d,L,1)
            # the backward of F.grid_sample is non-deterministic
            # See for details: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            out = F.grid_sample(feats, grid, **kwargs) # (gB,d,L,1)
            return rearrange(out, '... 1 -> ...') # (gB,d,L,1)-->(gB,d,L)

        def normalize_grid(arange, dim = 1, out_dim = -1):
            # normalizes 1d sequence to range of -1 to 1
            n = arange.shape[-1]
            return 2.0 * arange / max(n - 1, 1) - 1.0

        if self.offset_range_factor >= 0 and not self.no_off:
            offset = offset.tanh().mul(self.offset_range_factor) # 对偏移量进行缩放,获得\Dealta p^(g): (gB,L) * offset_range_factor

        if self.no_off:
            x_sampled = F.avg_pool1d(x, kernel_size=self.stride, stride=self.stride) # 没有偏移量的情况
        else:
            grid = torch.arange(offset.shape[-1], device = device) # 生成一个长度为L的有序序列: L
            vgrid = grid + offset # 将有序序列与偏移量相加，生成最终的采样位置: (gB,L)
            vgrid_scaled = normalize_grid(vgrid) # 对采样位置进行归一化，使其范围在 [-1, 1], 以便于后续的 grid_sample_1d: (gB,L)

            x_sampled = grid_sample_1d(
                group(x),
                vgrid_scaled,
                mode = 'bilinear', padding_mode = 'zeros', align_corners = False)[: ,: ,:L]  # 进行1D采样,使用双线性插值采样,输入输出shape相同: (B,C,L)-group->(gB,d,L);  (gB,d,L)-grid_sample_1d->(gB,d,L)

        if not self.no_off:
            x_sampled = rearrange(x_sampled ,'(b g) d n -> b (g d) n', g = self.n_groups) # 恢复shape: (gB,d,L)-->(B,C,L)
        q = q.reshape(B * self.n_heads, self.n_head_channels, L)  # 将q矩阵转换为多头注意力机制所需的形状: (B,C,L)-reshape->(Bk,h,L)  C=k*h; k是注意力头数量,h是每个头的通道数
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, L) # (Bk,h,L)
        if self.rpb:
            v = self.proj_v(x_sampled)
            v = (v + self.relative_position_bias_table).reshape(B * self.n_heads, self.n_head_channels, L) #如果使用了相对位置偏置(relative position bias, rpb), 则将相对位置偏置添加到 V中: (Bk,h,L)
        else:
            v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, L)

        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor # 计算缩放后的点积: (Bk,h,L)-einsum-(Bk,h,L)-->(Bk,h,h)

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1) # softmax: attention[0,0,:].sum() = 1

        out = torch.einsum('b i j , b j d -> b i d', attention, v) #使用注意力权重加权值向量，得到最终输出.  (Bk,h,h)-einsum-(Bk,h,L)-->(Bk,h,L)

        return self.proj_out(rearrange(out, '(b g) l c -> b c (g l)', b=B)) # (Bk,h,L)-rearrange->(B,L,C)-proj_out->(B,L,C)


class DeformAtten2D(nn.Module):
    '''
        max_offset (int): The maximum magnitude of the offset residue. Default: 14.
    '''
    def __init__(self, seq_len, d_model, n_heads, dropout, kernel=5, n_groups=4, no_off=False, rpb=True) -> None:
        super().__init__()
        self.offset_range_factor = kernel
        self.no_off = no_off
        self.f_sample = False
        self.seq_len = seq_len
        self.d_model = d_model # (512)
        self.n_groups = n_groups
        self.n_group_channels = self.d_model // self.n_groups
        self.n_heads = n_heads
        self.n_head_channels = self.d_model // self.n_heads
        self.n_group_heads = self.n_heads // self.n_groups
        self.scale = self.n_head_channels ** -0.5
        self.rpb = rpb

        self.proj_q = nn.Conv2d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Linear(self.d_model, self.d_model)
        kernel_size = kernel
        self.stride = 1
        pad_size = kernel_size // 2 if kernel_size != self.stride else 0
        self.proj_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kernel_size=kernel_size, stride=self.stride, padding=pad_size),
            nn.Conv2d(self.n_group_channels, 2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.scale_factor = self.d_model ** -0.5  # 1/np.sqrt(dim)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.d_model, self.seq_len, 1))
            trunc_normal_(self.relative_position_bias_table, std=.02)


    def forward(self, x, mask=None):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2) # B, C, H, W

        q = self.proj_q(x) # B, 1, H, W

        offset = self.proj_offset(q) # B, 2, H, W

        if self.offset_range_factor >= 0 and not self.no_off:
            offset = offset.tanh().mul(self.offset_range_factor)

        def create_grid_like(t, dim = 0):
            h, w, device = *t.shape[-2:], t.device

            grid = torch.stack(torch.meshgrid(
                torch.arange(w, device = device),
                torch.arange(h, device = device),
                indexing = 'xy'), dim = dim)

            grid.requires_grad = False
            grid = grid.type_as(t)
            return grid

        def normalize_grid(grid, dim = 1, out_dim = -1):
            # normalizes a grid to range from -1 to 1
            h, w = grid.shape[-2:]
            grid_h, grid_w = grid.unbind(dim = dim)

            grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
            grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

            return torch.stack((grid_h, grid_w), dim = out_dim)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        else:
            grid =create_grid_like(offset)
            vgrid = grid + offset
            vgrid_scaled = normalize_grid(vgrid)
            # the backward of F.grid_sample is non-deterministic
            x_sampled = F.grid_sample(
                x,
                vgrid_scaled,
                mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, :H, :W]

        if not self.no_off:
            x_sampled = rearrange(x_sampled, '(b g) c h w -> b (g c) h w', g=self.n_groups)
        q = q.reshape(B * self.n_heads, H, W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, H, W)
        if self.rpb:
            v = self.proj_v(x_sampled)
            v = (v + self.relative_position_bias_table).reshape(B * self.n_heads, H, W)
        else:
            v = self.proj_v(x_sampled).reshape(B * self.n_heads, H, W)

        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)

        out = torch.einsum('b i j , b j d -> b i d', attention, v)

        return self.proj_out(out.reshape(B, H, W, C))


class CrossDeformAttn(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, dropout, droprate,
                 n_days=1, window_size=4, patch_len=7, stride=3, no_off=False) -> None:
        super().__init__()
        self.n_days = n_days
        self.seq_len = seq_len
        # 1d size: B*n_days, subseq_len, C
        # 2d size: B*num_patches, 1, patch_len, C
        self.subseq_len = seq_len // n_days + (1 if seq_len % n_days != 0 else 0)
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = num_patches(self.seq_len, self.patch_len, self.stride)

        self.layer_norm = nn.LayerNorm(d_model)

        # 1D
        self.ff1 = nn.Linear(d_model, d_model, bias=True)
        self.ff2 = nn.Linear(self.subseq_len, self.subseq_len, bias=True)
        # Deform attention
        self.deform_attn = DeformAtten1D(self.subseq_len, d_model, n_heads, dropout, kernel=window_size, no_off=no_off)
        self.attn_layers1d = nn.ModuleList([self.deform_attn])

        self.mlps1d = nn.ModuleList(
            [
                MLP([d_model, d_model], final_relu=True, drop_out=0.0) for _ in range(len(self.attn_layers1d))
            ]
        )
        self.drop_path1d = nn.ModuleList(
            [
                DropPath(droprate) if droprate > 0.0 else nn.Identity() for _ in range(len(self.attn_layers1d))
            ]
        )
        #######################################
        # 2D
        d_route = 1
        self.conv_in = nn.Conv2d(1, d_route, kernel_size=1, bias=True)
        self.conv_out = nn.Conv2d(d_route, 1, kernel_size=1, bias=True)
        self.deform_attn2d = DeformAtten2D(self.patch_len, d_route, n_heads=1, dropout=dropout, kernel=window_size,
                                           n_groups=1, no_off=no_off)
        self.write_out = nn.Linear(self.num_patches * self.patch_len, self.seq_len)

        self.attn_layers2d = nn.ModuleList([self.deform_attn2d])

        self.mlps2d = nn.ModuleList(
            [
                MLP([d_model, d_model], final_relu=True, drop_out=0.0) for _ in range(len(self.attn_layers2d))
            ]
        )
        self.drop_path2d = nn.ModuleList(
            [
                DropPath(droprate) if droprate > 0.0 else nn.Identity() for _ in range(len(self.attn_layers2d))
            ]
        )

        self.fc = nn.Linear(2 * d_model, d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        n_day = self.n_days
        B, L, C = x.shape

        x = self.layer_norm(x)

        padding_len = (n_day - (L % n_day)) % n_day
        x_padded = torch.cat((x, x[:, [0], :].expand(-1, padding_len, -1)), dim=1) # (B,L,C)
        x_1d = rearrange(x_padded, 'b (seg_num ts_d) d_model -> (b ts_d) seg_num d_model', ts_d=n_day) #(B,L,C)--rearrange-->(B,L,C)
        # attn on 1D
        for d, attn_layer in enumerate(self.attn_layers1d):
            x0 = x_1d
            x_1d = attn_layer(x_1d) # (B,L,C)-->(B,L,C)
            x_1d = self.drop_path1d[d](x_1d) + x0
            x0 = x_1d
            x_1d = self.mlps1d[d](self.layer_norm(x_1d))
            x_1d = self.drop_path1d[d](x_1d) + x0
        x_1d = rearrange(x_1d, '(b ts_d) seg_num d_model -> b (seg_num ts_d) d_model', ts_d=n_day)[:, :L, :]   # (B,L,C)--rearrange-->(B,L,C)

        # Patch attn on 2D
        x_unfold = x.unfold(dimension=-2, size=self.patch_len, step=self.stride)
        x_2d = rearrange(x_unfold, 'b n c l -> (b n) l c').unsqueeze(-3)
        x_2d = rearrange(x_2d, 'b c h w -> b h w c')
        for d, attn_layer in enumerate(self.attn_layers2d):
            x0 = x_2d
            x_2d = attn_layer(x_2d)
            x_2d = self.drop_path2d[d](x_2d) + x0
            x0 = x_2d
            x_2d = self.mlps2d[d](self.layer_norm(x_2d.permute(0, 1, 3, 2))).permute(0, 1, 3, 2)
            x_2d = self.drop_path2d[d](x_2d) + x0
        x_2d = rearrange(x_2d, 'b h w c -> b c h w')
        x_2d = rearrange(x_2d, '(b n) 1 l c -> b (n l) c', b=B)
        x_2d = self.write_out(x_2d.permute(0, 2, 1)).permute(0, 2, 1)

        x = torch.concat([x_1d, x_2d], dim=-1)
        x = self.fc(x)

        return x



if __name__ == '__main__':
    # (B,L,D)
    x1 = torch.randn(1,196,64).to(device)

    Model = CrossDeformAttn(seq_len=196, d_model=64, n_heads=8, dropout=0., droprate=0.,n_days=1).to(device)

    out = Model(x1) # (B,L,D)-->(B,L,D)
    print(out.shape)