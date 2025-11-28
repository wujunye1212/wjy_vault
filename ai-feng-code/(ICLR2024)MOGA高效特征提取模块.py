import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import  DropPath
# https://arxiv.org/pdf/2211.03295
# https://github.com/Westlake-AI/MogaNet.
''''
来自ICLR 2024 CCF-A顶会
在MogaNet模块中，Moga模块的主要原理是通过多阶的方式来进行特征的聚合，
并且有效地捕获不同层次的上下文信息。
Moga模块的作用是改善传统卷积神经网络在上下文特征编码方面的局限性，
尤其是应对现代深度神经网络在捕获低阶与高阶交互方面的挑战。

Moga模块的原理：
多阶交互机制：Moga模块通过设计多阶的深度可分离卷积（DWConv）来捕获局部与全局的上下文信息。
            它将输入特征进行多阶分解，通过不同的膨胀率（dilation rates）分别处理低阶、中阶和高阶的特征交互，
            从而增强特征的表达能力。

特征门控机制：Moga模块通过门控机制，自适应地聚合不同尺度的特征信息。
     采用了SiLU激活函数，它不仅具有Sigmoid的门控效果，还能够使得网络训练更加稳定。
     该模块将通过门控分支和上下文分支的特征信息相乘进行聚合，从而使得网络更关注有用的特征信息并忽略冗余部分。

Moga模块的作用：
通过多阶特征的捕获和解构，Moga模块能够生成更加丰富且有用的上下文信息，使得网络在图像分类、目标检测等任务中表现出色。
通过门控机制和特征解构，Moga模块有效减少了特征中的冗余信息，提升了计算效率并保持较高的表达能力。
该模块能自适应地根据输入特征选择合适的交互级别，从而为不同任务提供更具针对性的特征表达。

适用于：图像分类，目标检测、实例分割、语义分割，关键点检测等所有CV任务通用的即插即用模块
'''
class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = nn.GELU()

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,],
                ):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation.

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for Spatial Block.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x


class MogaBlock(nn.Module):
    """A block of MogaNet.

    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_type (str): The activation type for projections and FFNs.
            Defaults to 'GELU'.
        norm_cfg (str): The type of normalization layer. Defaults to 'BN'.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for the gating branch.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                ):
        super(MogaBlock, self).__init__()
        self.out_channels = embed_dims

        self.norm1 = nn.BatchNorm2d(embed_dims)

        # spatial attention
        self.attn = MultiOrderGatedAggregation(
            embed_dims,
            attn_dw_dilation=attn_dw_dilation,
            attn_channel_split=attn_channel_split,
            attn_act_type=attn_act_type,
            attn_force_fp32=attn_force_fp32,
        )
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(embed_dims)

        # channel MLP
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(  # DWConv + Channel Aggregation FFN
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_type=act_type,
            ffn_drop=drop_rate,
        )

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        # spatial
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x

if __name__ == '__main__':

    input = torch.randn(1, 64, 32, 32)
    MOGA = MultiOrderGatedAggregation(64)
    MogaBlock = MogaBlock(64)
    output = MOGA(input)
    print(' MOGA_input_size:', input.size())
    print(' MOGA_output_size:', output.size())

    output = MogaBlock(input)
    print('MogaBlock input_size:', input.size())
    print('MogaBlock output_size:', output.size())

