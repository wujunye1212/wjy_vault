import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm2d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer2D, self).__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm1d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer1D, self).__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim = 64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(d_model, 3* state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim * 3
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0)
        self.hz_proj = ConvLayer1D(d_model, 2 * self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x):
        batch, _, L = x.shape  # (B,C,L)
        H = int(math.sqrt(L)) # H

        BCdt = self.dw(self.BCdt_proj(x).view(batch, -1, H, H)).flatten(2) # 做线性投影得到B,C,Delta: (B,C,L)--proj-->(B,3C,L)--view-->(B,3C,H,H)--dw-->(B,3C,H,H)--flatten-->(B,3C,L)
        B, C, dt = torch.split(BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1) # (B,3C,L)--split-->B: (B,C,L), C: (B,C,L), dt: (B,C,L)
        A = (dt + self.A.view(1, -1, 1)).softmax(-1) # 离散化A: (B,C,L) + (C,)-view->(1,C,1) == (B,C,L)

        AB = (A * B) # 通过矩阵点乘得到权重: (B,C,L) * (B,C,L)
        h = x @ AB.transpose(-2, -1) # 把L个token信息聚合到C个全局隐藏状态中: (B,C,L) @ (B,L,C) == (B,C,C)

        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1) # 在隐藏状态上做一次通道线性投影,得到两路: (B,C,C)--proj-->(B,2C,C)--split-->h: (B,C,C), z: (B,C,C)
        h = self.out_proj(h * self.act(z) + h * self.D) # σ(z)做门控, 对h进行强调或抑制; 在这里还额外使用了一个参数D进行调整: (B,C,C) * (B,C,C) + (B,C,C) * (1,) == (B,C,C)
        y = h @ C  # 输出矩阵调整: (B,C,C) @ (B,C,L) == (B,C,L)

        y = y.view(batch, -1, H, H).contiguous()  # 转变为你想要的shape进行输出,如果你是一维序列则不需要进行这步变化:  (B,C,L)-view->(B,C,H,H)
        return y, h


if __name__ == '__main__':
    # (B,C,N)
    x1 = torch.randn(1,64,196).to(device)

    # 超参数
    B, C, N = x1.size()

    Model = HSMSSD(d_model=C, ssd_expand=1,state_dim=C).to(device)

    y, h = Model(x1)
    print(y.shape)