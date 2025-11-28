from einops import rearrange
import torch
from torch.nn import functional as F
from torch import nn
# https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_DNF_Decouple_and_Feedback_Network_for_Seeing_in_the_Dark_CVPR_2023_paper.pdf
# https://github.com/Srameo/DNF/tree/main
'''
CVPR 2023顶会

本文介绍了一个新颖的网络架构——Decouple and Feedback Network (DNF)，
其目标是解决现有方法在基于RAW图像的低光图像增强任务中的性能瓶颈。
为了应对单阶段方法的域歧义问题以及多阶段方法的信息丢失问题，
DNF网络提出了两个主要的创新点：域特定任务解耦 和去噪反馈机制 。
DNF网络框架由：CID ，MCC，GFM

CID 模块的作用： CID 模块负责在 RAW 域内执行独立的去噪任务。由于 RAW 图像中的噪声通常是信号无关的，
并且各个通道间的噪声分布独立，因此 CID 模块使用了深度卷积（7x7卷积核）来移除通道独立的噪声。
每个 CID 模块独立处理各通道的噪声，并且通过残差结构进一步增强去噪效果。该模块的引入保证了 RAW 图像的去噪精度。

MCC 模块的作用： MCC 模块专注于 RAW 到 sRGB 的颜色转换任务。在图像信号处理 (ISP) 流水线中，
颜色转换通常通过通道级的矩阵变换实现。MCC 模块通过1x1卷积层和3x3深度卷积层生成Q、K和V矩阵，
进行颜色空间转换和局部细节的优化，确保最终 sRGB 图像的颜色校正效果。

GFM 模块的作用： GFM 模块用于实现特征级别的信息反馈，将 RAW 解码器输出的去噪特征重新注入到 RAW 编码器中，
以改善去噪效果。通过门控机制，GFM 能够自适应地融合反馈特征和初始去噪特征，进而使得网络在噪声与细节之间进行有效区分，
从而减少去噪过程中的细节丢失。

总结：
CID 模块通过独立通道的去噪提升了 RAW 图像中的去噪能力，
MCC 模块则通过矩阵变换实现了高效的颜色校正，
GFM 模块通过门控机制进行特征融合，解决了传统多阶段方法中的信息丢失问题。实现了更高效的低光图像增强。

即插即用模块适用于：图像增强，低光图像增强，图像去噪，图像恢复，低光目标检测，低光图像分割等所有CV任务通用模块
'''
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
# CI
class DConv7(nn.Module):
    def __init__(self, f_number, padding_mode='reflect') -> None:
        super().__init__()
        self.dconv = nn.Conv2d(f_number, f_number, kernel_size=7, padding=3, groups=f_number, padding_mode=padding_mode)

    def forward(self, x):
        return self.dconv(x)

# Post-CI
class MLP(nn.Module):
    def __init__(self, f_number, excitation_factor=2) -> None:
        super().__init__()
        self.act = nn.GELU()
        self.pwconv1 = nn.Conv2d(f_number, excitation_factor * f_number, kernel_size=1)
        self.pwconv2 = nn.Conv2d(f_number * excitation_factor, f_number, kernel_size=1)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x

class CID(nn.Module):
    def __init__(self, f_number, padding_mode) -> None:
        super().__init__()
        self.channel_independent = DConv7(f_number, padding_mode)
        self.channel_dependent = MLP(f_number, excitation_factor=2)

    def forward(self, x):
        return self.channel_dependent(self.channel_independent(x))

class MCC(nn.Module):
    def __init__(self, f_number, num_heads, padding_mode, bias=False) -> None:
        super().__init__()
        self.norm = LayerNorm(f_number, eps=1e-6, data_format='channels_first')

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.pwconv = nn.Conv2d(f_number, f_number * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(f_number * 3, f_number * 3, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=f_number * 3)
        self.project_out = nn.Conv2d(f_number, f_number, kernel_size=1, bias=bias)
        self.feedforward = nn.Sequential(
            nn.Conv2d(f_number, f_number, 1, 1, 0, bias=bias),
            nn.GELU(),
            nn.Conv2d(f_number, f_number, 3, 1, 1, bias=bias, groups=f_number, padding_mode=padding_mode),
            nn.GELU()
        )

    def forward(self, x):
        attn = self.norm(x)
        _, _, h, w = attn.shape

        qkv = self.dwconv(self.pwconv(attn))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.feedforward(out + x)
        return out

class GFM(nn.Module):
    def __init__(self, in_channels, feature_num=2, bias=True, padding_mode='reflect', **kwargs) -> None:
        super().__init__()
        self.feature_num = feature_num

        hidden_features = in_channels * feature_num
        self.pwconv = nn.Conv2d(hidden_features, hidden_features * 2, 1, 1, 0, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=hidden_features * 2)
        self.project_out = nn.Conv2d(hidden_features, in_channels, kernel_size=1, bias=bias)
        self.mlp = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)

    def forward(self, *inp_feats):
        assert len(inp_feats) == self.feature_num
        shortcut = inp_feats[0]
        x = torch.cat(inp_feats, dim=1)
        x = self.pwconv(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return self.mlp(x + shortcut)

# 输入 N C D H W,  输出 N C D H W
if __name__ == '__main__':
    # 创建一个简单的输入特征图
    input1 = torch.randn(2, 64, 32, 32)
    input2 = torch.randn(2, 64, 32, 32)
    # 创建一个 GFM 实例
    GFM = GFM(in_channels=64)
    # 将两个输入特征图传递给 GFM 模块
    output = GFM(input1, input2)
    # 打印输入和输出的尺寸
    print(f"input 1 shape: {input1.shape}")
    print(f"input 2 shape: {input2.shape}")
    print(f"output shape: {output.shape}")