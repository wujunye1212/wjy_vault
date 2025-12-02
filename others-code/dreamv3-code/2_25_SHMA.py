import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import trunc_normal_



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    # split the channel, however, it is slower
    # H_split = x.split(window_size, dim = 2)
    # HW_split = []
    # for split in H_split:
    #     HW_split += split.split(window_size, dim=-1)
    # windows = torch.cat(HW_split, dim=0)
    # return windows

    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size) # (B,C,H,W)-->(B,C,H/WS,WS,W/WS,WS)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size) # (B,C,H/WS,WS,W/WS,WS)--permute-->(B,H/WS,W/WS,C,WS,WS)-view->(B*H/WS*W/WS,C,WS,WS)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    _, C, _, _ = windows.shape  # (B*H/WS*W/WS,C,WS,WS)
    # split the channel, however, it is slower
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # HW_split = windows.split(B, dim = 0)
    # H_split = []
    # split_size = W // window_size
    # for i in range(H // window_size):
    #     H_split.append(torch.cat(HW_split[i * split_size:(i + 1) * split_size], dim=2))
    # x = torch.cat(H_split, dim=-1)
    # return x

    B = int(windows.shape[0] / (H * W / window_size / window_size)) # (B*H/WS*W/WS) / (HW/WS/WS) == B
    x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size) # (B*H/WS*W/WS,C,WS,WS)--view-->(B,H/WS,W/WS,C,WS,WS)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W) # (B,H/WS,W/WS,C,WS,WS)--permute-->(B,C,H/WS,W/WS,WS,WS)--view-->(B,C,H,W)
    return x


class WindowPartion(nn.Module):
    def __init__(self,
                 window_size=0,
                 **kargs):
        super().__init__()
        assert window_size > 0
        self.window_size = window_size

    def forward(self, x):
        if type(x) is tuple:
            x = x[0]
        B, C, H, W = x.shape
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0))
        Ho, Wo = H, W
        _, _, Hp, Wp = x.shape
        x = window_partition(x, self.window_size)
        return x, (Ho, Wo, Hp, Wp, pad_r, pad_b)


class WindowReverse(nn.Module):
    def __init__(self,
                 window_size=0,
                 **kargs):
        super().__init__()
        assert window_size > 0
        self.window_size = window_size

    def forward(self, x):
        x, (Ho, Wo, Hp, Wp, pad_r, pad_b) = x[0], x[1]
        x = window_reverse(x, self.window_size, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :Ho, :Wo].contiguous()
        return x, (Ho, Wo, Hp, Wp, pad_r, pad_b)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class SHMA(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=1,
            attn_drop=0.,
            fused_attn=False,
            ratio=4,
            q_kernel=1,
            kv_kernel=1,
            kv_stride=1,
            head_dim_reduce_ratio=2,
            window_size=0,
            sep_v_gate=False,
            **kwargs,
    ):
        super().__init__()
        mid_dim = int(dim * ratio) # R*C
        dim_attn = dim // head_dim_reduce_ratio # C/K
        self.num_heads = num_heads
        self.dim_head = dim_attn // num_heads
        self.v_dim_head = mid_dim // self.num_heads
        self.scale = self.dim_head ** -0.5
        self.fused_attn = fused_attn

        self.q = Conv2d_BN(dim, dim_attn, q_kernel, stride=1, pad=q_kernel // 2)
        self.k = Conv2d_BN(dim, dim_attn, kv_kernel, stride=kv_stride, pad=kv_kernel // 2)
        self.gate_act = nn.Sigmoid()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Conv2d_BN(mid_dim, dim, 1)
        self.window_size = window_size
        #self.block_index = kwargs['block_index']
        self.kv_stride = kv_stride
        self.sep_v_gate = sep_v_gate
        self.v_gate = Conv2d_BN(dim, 2 * mid_dim, kv_kernel, stride=kv_stride, pad=kv_kernel // 2)

    def forward(self, x, attn_mask=None):
        B, C, H, W = x.shape
        if self.window_size:
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size # 可以整除的情况下, 为0
            pad_b = (self.window_size - H % self.window_size) % self.window_size #
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0))
            Ho, Wo = H, W
            _, _, Hp, Wp= x.shape
            x = window_partition(x, self.window_size) # (B,C,H,W)-->(B*H/WS*W/WS,C,WS,WS)
            B, C, H, W = x.shape
        v, gate = self.gate_act(self.v_gate(x)).chunk(2, dim=1) # (B,C,H,W)-gate->(B,2*R*C,H,W)-gate_act->(B,2*R*C,H,W);  v:(B,R*C,H,W), gate:(B,R*C,H,W)
        q_short = self.q(x) # (B,C,H,W)--q-->(B,C/K,H,W)
        q = q_short.flatten(2) # (B,C/K,H,W)--flatten-->(B,C/K,HW)
        k = self.k(x).flatten(2) # (B,C,H,W)--k-->(B,C/K,H,W)--flatten-->(B,C/K,HW)

        v = v.flatten(2) # (B,R*C,H,W)--flatten-->(B,R*C,HW)
        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(-1, -2).contiguous(),
                k.transpose(-1, -2).contiguous(),
                v.transpose(-1, -2).contiguous(),
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            ).transpose(-1, -2).reshape(B, -1, H, W)
        else:
            q = q * self.scale # 对q进行缩放
            attn = q.transpose(-2, -1) @ k # (B,HW,C/K) @ (B,C/K,HW) == (B,HW,HW)
            if attn_mask is not None:
                # NOTE: assumes mask is float and in correct shape
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1) # (B,HW,HW)
            attn = self.attn_drop(attn) # (B,HW,HW)
            x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W) # (B,R*C,HW) @ (B,HW,HW) == (B,R*C,HW);  (B,R*C,HW)--view-->(B,R*C,H,W)

        x = x * gate # (B,R*C,H,W) * (B,R*C,H,W) == (B,R*C,H,W)
        x = self.proj(x) # (B,R*C,H,W)--proj-->(B,C,H,W)
        if self.window_size:
            x = window_reverse(x, self.window_size, Hp, Wp) # 事实上, x的真正shape是(B*H/WS*W/WS,C,WS,WS); (B*H/WS*W/WS,C,WS,WS)--window_reverse-->(B,C,H,W)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :Ho, :Wo].contiguous()
        return x



if __name__ == '__main__':
    # block_types = ['ConvBlock_k7_r4'] * 2 + ['ConvBlock_k7_r4'] * 2 + ['ConvBlock_k7_r4'] * 9 + \
    #               ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1_ws16_wsp1_fa1', 'FFN2d_r3'] + \
    #               ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1_fa1', 'FFN2d_r3'] * 2 + \
    #               ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1_ws16_wre1_fa1', 'FFN2d_r3'] + ['ConvBlock_k7_r4'] + \
    #               ['RepCPE_k3', 'SHMABlock_r1_hdrr4_act0_nh1_fa1', 'FFN2d_r3'] * 2
    # print(block_types)
    # (B,C,H,W)
    x1 = torch.randn(1, 64, 224, 224)
    B, C, H, W = x1.size()

    # 定义 SHMA
    Model = SHMA(dim=64, window_size=16)

    # 执行 SHMA
    out = Model(x1)
    print(out.shape)