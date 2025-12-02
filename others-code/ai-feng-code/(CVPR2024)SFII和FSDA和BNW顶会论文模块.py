# https://github.com/Xiaofeng-life/SFSNiD/blob/master/methods/MyNightDehazing/SFSNiD.py#L252
# https://arxiv.org/pdf/2403.18548
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
import warnings
warnings.filterwarnings('ignore')
'''
CVPR 2024顶会论文
本文SFII、BNM 和 FSDA 三个即插即用模块的作用：
1. SFII (Spatial and Frequency Information Interaction) 模块
SFII 模块用于结合空间域和频率域的信息处理，应对夜间去雾任务中的各种复杂失真（如雾霾、眩光和噪声等）。
作用：
解决了图像中的局部化、耦合和频率不一致的问题。
在频率域内动态地过滤和聚合多通道的幅度谱和相位谱，并通过局部感知机制在空间域进行整合。
特点：
实现了空间和频率域之间的双域交互，增强了特征提取和图像重建的效果。

2. BNM (Bidomain Nonlinear Mapping) 模块
BNM 模块旨在通过双域的非线性映射提高模型的表达能力。
作用：
在频率域和空间域之间建立复杂的交互，用于学习非线性映射关系。
利用残差连接来确保特征传递，并在两个域内执行独立的非线性映射。
特点：
包含一个用于频率域处理的 FSDA 模块和一个用于空间域处理的残差块。
通过这两者的融合，提升模型的非线性表达能力。

3. FSDA (Frequency Spectrum Dynamic Aggregation) 模块
FSDA 模块主要在频率域中处理不同频率特性的失真（如低频的雾霾和高频的噪声）。
作用：
动态地过滤和聚合多通道的频率谱信息，消除噪声和眩光，同时保留图像的细节。
通过卷积操作对不同通道的频率谱进行加权，并将结果反馈至空间域，实现双域信息的融合。
这些模块共同作用，提升了模型在夜间复杂场景中的去雾性能，同时确保了去雾后的图像具有真实的亮度表现。

这三个即插即用模块适用于：
图像去雾，图像去噪，超分图像重建，图像恢复，图像增强，目标检测，图像分割，暗光增强，细节边缘增强等所有计算机视觉CV任务通用模块

'''
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x
class BidomainNonlinearMapping_SinglePath(nn.Module):

    def __init__(self, in_nc):
        super(BidomainNonlinearMapping_SinglePath, self).__init__()
        self.frequency_process = Frequency_Spectrum_Dynamic_Aggregation(in_nc)

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')

        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')

        return x_freq_spatial + x
class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Sequential(BidomainNonlinearMapping_SinglePath(in_nc=dim), nn.Conv2d(dim, dim, 1))
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.Q = nn.Sequential(BidomainNonlinearMapping_SinglePath(in_nc=dim), nn.Conv2d(dim, dim, 1))
            self.K = nn.Sequential(BidomainNonlinearMapping_SinglePath(in_nc=dim), nn.Conv2d(dim, dim, 1))
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        V = self.V(X)

        if self.use_attn:
            # QK = self.QK(X)
            Q = self.Q(X)
            K = self.K(X)
            QKV = torch.cat([Q, K, V], dim=1)

            # shift
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            # partition windows
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

            attn_windows = self.attn(qkv)

            # merge windows
            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # reverse cyclic shift
            out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
            attn_out = out.permute(0, 3, 1, 2)

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)

        else:
            if self.conv_type == 'Conv':
                out = self.conv(X)  # no attention and use conv, no projection
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))

        return out
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
class ResBlock_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock_Conv, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.trans_layer = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans_layer(out)
        out = self.conv2(out)
        return out + x
class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = ResBlock_Conv(in_channel=nc, out_channel=nc)

    def forward(self, x):
        yy = self.block(x)
        return yy
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class Frequency_Spectrum_Dynamic_Aggregation(nn.Module):
    def __init__(self, nc):
        super(Frequency_Spectrum_Dynamic_Aggregation, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        ori_mag = torch.abs(x)
        ori_pha = torch.angle(x)
        mag = self.processmag(ori_mag)
        mag = ori_mag + mag
        pha = self.processpha(ori_pha)
        pha = ori_pha + pha
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out
class BidomainNonlinearMapping(nn.Module):
    def __init__(self, in_nc):
        super(BidomainNonlinearMapping, self).__init__()
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = Frequency_Spectrum_Dynamic_Aggregation(in_nc)
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 1, 1, 0)
    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')

        xcat = torch.cat([x, x_freq_spatial], 1)
        x_out = self.cat(xcat)

        return x_out
class SFII(nn.Module):
    def __init__(self, dim, num_heads=2, network_depth=6, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm
        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)
        self.fft_block = BidomainNonlinearMapping(in_nc=dim)

    def forward(self, x):
        B, C, H, W = x.size()
        identity = x
        if self.use_attn:
            x = x.view(B, C, H * W) #还可以使用它x.reshape(B,C,H*W)
            x = x.transpose(1, 2) #还可以使用它x.permute(0,2,1)
            x = self.norm1(x)
            x = x.transpose(1, 2)
            x = x.view(B, C, H, W)
        x = self.attn(x)
        x = identity + x
        temp = x
        x = self.fft_block(x)
        x = x + temp
        return x


# 输入 B C H W   输出 B C H W
if __name__ == '__main__':
    # 创建输入张量
    input = torch.randn(1, 32,64,64)
    sfii = SFII(32)
    output = sfii(input)
    print('SFII_input_size:', input.size())
    print('SFII_output_size:', output.size())

    BNW = BidomainNonlinearMapping(32)
    output = BNW(input)
    print('BNW_input_size:', input.size())
    print('BNW_output_size:', output.size())

    FSDA = Frequency_Spectrum_Dynamic_Aggregation(32)
    output = FSDA(input)
    print('FSDA_input_size:', input.size())
    print('FSDA_output_size:', output.size())


