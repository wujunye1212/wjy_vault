import torch
import torch.nn as nn
import numbers
from einops import rearrange
# https://arxiv.org/pdf/2312.08866
# https://github.com/haoshao-nku/medical_seg/blob/master/mmsegmentation/mmseg/models/decode_heads/MCANet.py

'''
即插即用模块：MSCA 多尺度交叉轴注意力模块  2023 Arxiv
本提出了一种新的医学图像分割注意力模块，名为多尺度交叉轴注意力模块（MSCA）。
该方法主要解决两个关键问题：如何有效捕捉多尺度信息以及建立像素间的长距离依赖关系，
这在医学图像分割中至关重要，特别是针对尺寸和形状差异较大的情况。
MSCA模块的作用：
1.多尺度信息捕捉：在医学图像分割任务中，分割样本的尺寸和形状变化多样。
为了解决这个问题，论文提出在每个轴向注意力路径中使用不同尺寸的条形卷积核，从而提高空间信息编码的效率。
2.交叉轴注意力机制：MSCA 模块通过同时计算两个平行轴向注意力之间的交叉注意力来更好地捕捉全局信息，
而不仅仅是沿水平和垂直方向顺序连接。这种设计能够更有效地利用多尺度特征，帮助网络识别边界模糊的目标区域。
3.MSCA 模块的效率：MSCA 模块基于高效的轴向注意力机制，但通过引入多尺度卷积进一步增强其性能，
而不会显著增加计算复杂度。在多个医学图像分割任务上表现优异，如皮肤病、细胞核、腹部多器官和息肉的分割。

MSCA模块的主要作用在于同时捕捉多尺度信息和建立长距离依赖关系。
它通过双重交叉注意力的机制使模型能够从不同尺度和方向捕捉全局上下文信息。
这种设计特别适用于医学图像分割中病灶区域尺寸变化大、边界模糊的情况，显著提高了分割精度。
MCA适用于：医学图像分割，实例分割，语义分割，目标检测等所有计算机视觉CV任务通用的注意力模块
'''
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight
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
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class MSCAttention(nn.Module):
    def __init__(self, dim, num_heads=8, LayerNorm_type='WithBias'):
        super(MSCAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.norm1(x)
        attn_00 = self.conv0_1(x1)
        attn_01 = self.conv0_2(x1)
        attn_10 = self.conv1_1(x1)
        attn_11 = self.conv1_2(x1)
        attn_20 = self.conv2_1(x1)
        attn_21 = self.conv2_2(x1)
        out1 = attn_00 + attn_10 + attn_20
        out2 = attn_01 + attn_11 + attn_21
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out3) + self.project_out(out4) + x
        return out

    # 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input = torch.randn(1, 64, 32, 32)
    # 创建一个 MSCA 模块实例
    msca = MSCAttention(dim=64)
    # 执行前向传播
    output = msca(input)
    # 打印输入和输出的形状
    print('input_size:',input.size())
    print('output_size:',output.size())