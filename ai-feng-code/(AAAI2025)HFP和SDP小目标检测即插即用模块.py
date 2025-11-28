# Code Structure of HS-FPN (https://arxiv.org/abs/2412.10116)
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT  # pip install torch_dct
from einops import rearrange
#看Ai缝合怪b站视频：2025.7.17 更新的视频
class DctSpatialInteraction(nn.Module):
    def __init__(self,
                 in_channels,
                 ratio,
                 isdct=True):
        super(DctSpatialInteraction, self).__init__()
        self.ratio = ratio
        self.isdct = isdct  # true when in p1&p2 # false when in p3&p4
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
                *[nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)]
            )

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)
        dct = idct * weight  # filter out low-frequency features
        dct_ = DCT.idct_2d(dct, norm='ortho')  # generate spatial mask
        return x * dct_

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


# ------------------------------------------------------------------#
# Channel Path of HFP
# Only p1&p2 use dct to extract high_frequency response
# ------------------------------------------------------------------#
class DctChannelInteraction(nn.Module):
    def __init__(self,
                 in_channels,
                 patch,
                 ratio,
                 isdct=True
                 ):
        super(DctChannelInteraction, self).__init__()
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, 1, groups=32)],
        )
        self.channel2x1 = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, 1, groups=32)],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        if not self.isdct:  # true when in p1&p2 # false when in p3&p4
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))  # 2025 03 15 szc
            return x * torch.sigmoid(self.channel2x1(channel))

        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)
        dct = idct * weight  # filter out low-frequency features
        dct_ = DCT.idct_2d(dct, norm='ortho')

        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        # channel = torch.cat([self.channel1x1(aavgp), self.channel1x1(amaxp)], dim = 1) # TODO: The values of aavgp and amaxp appear to be on different scales. Add is a better choice instead of concate.
        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)  # 2025 03 15 szc
        return x * torch.sigmoid(self.channel2x1(channel))

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

# High Frequency Perception Module HFP
# ------------------------------------------------------------------#
class HFP(nn.Module):
    def __init__(self,
                 in_channels,
                 ratio=(0.25, 0.25),
                 patch=(8, 8),
                 isdct=True):
        super(HFP, self).__init__()
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)
        self.out = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
              nn.GroupNorm(32, in_channels)]
        )

    def forward(self, x):
        spatial = self.spatial(x)  # output of spatial path
        channel = self.channel(x)  # output of channel path
        return self.out(spatial + channel)
# ------------------------------------------------------------------#
# Spatial Dependency Perception Module SDP
# ------------------------------------------------------------------#
class SDPFusion(nn.Module):
    def __init__(self,
                 dim=256,
                 patch=None,
                 inter_dim=None
               ):
        super(SDPFusion, self).__init__()
        self.dim = dim
        self.inter_dim=inter_dim
        if self.inter_dim == None:
            self.inter_dim = dim
        self.conv_q = nn.Sequential(*[nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.conv_k = nn.Sequential(*[nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.softmax = nn.Softmax(dim=-1)
        self.patch_size = (patch,patch)
        self.conv1x1 = Conv(self.dim,self.inter_dim,1)
    def forward(self, x_low, x_high):

        b_, _, h_, w_ = x_low.size()
        q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=self.patch_size[0],p2=self.patch_size[1])
        q = q.transpose(1, 2)  # 1,4096,128
        k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=self.patch_size[0],p2=self.patch_size[1])
        attn = torch.matmul(q, k)  # 1, 4096, 1024
        attn = attn / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1, 2)  # 1, 1024, 128
        output = torch.matmul(attn, v)  # 1, 4096, 128
        output = rearrange(output.transpose(1, 2).contiguous(), '(b h w) c (p1 p2) -> b c (h p1) (w p2)',p1=self.patch_size[0], p2=self.patch_size[1], h=h_ // self.patch_size[0],w=w_ // self.patch_size[1])
        if self.dim != self.inter_dim:
            x_low = self.conv1x1(x_low)
        return output + x_low
# AAAI2025 HFP模块的二次创新，HLPFM在我的二次创新模块交流群，可以直接去跑实验发小论文！
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input = torch.randn(1, 32, 64, 64)
    # 创建 HFP 模块
    hfp = HFP(in_channels=32)  #第一个模块
    # 将输入图像传入HFP模块进行处理
    output = hfp(input)
    # 输出结果的形状
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新_HFP_input_size:', input.size())
    print('Ai缝合即插即用模块永久更新_HFP_output_size:', output.size())

    # 定义输入张量的形状为 B, C, H, W
    input1= torch.randn(1, 64, 128, 128)
    input2 = torch.randn(1, 64, 128, 128)
    # 创建 SDP 模块
    # 第二个模块 ,第一个参数64是输入通道数，第二个参数8是patch_size参数，第三参数32代表输出通道数
    sdp= SDPFusion(64,8,32)
    # 将输入图像传入 SDP 模块进行处理
    output = sdp(input1,input2)
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新-SDP_input_size:', input1.size())
    print('Ai缝合即插即用模块永久更新-SDP_output_size:', output.size())
# AAAI2025 HFP模块的二次创新，HLPFM在我的二次创新模块交流群，可以直接去跑实验发小论文！

