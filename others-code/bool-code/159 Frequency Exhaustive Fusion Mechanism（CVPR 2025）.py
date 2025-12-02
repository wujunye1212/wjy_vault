import torch
import torch.nn as nn
import numbers
from einops import rearrange
import math
import torch_dct as dct
# pip install torch_dct

"""    
    论文地址：https://arxiv.org/pdf/2412.16645v1
    论文题目：Complementary Advantages: Exploiting Cross-Field Frequency Correlation for NIR-Assisted Image Denoising （CVPR 2025）
    中文题目：互补优势：利用跨域频率相关性实现近红外辅助图像去噪 （CVPR 2025）
    讲解视频：https://www.bilibili.com/video/BV1rz7tz3ETa/
        频域全面融合机制（Frequency Exhaustive Fusion Mechanism，FEFM）：
            实际意义：①跨域特征失衡：传统方法仅强化公共特征，忽略 NIR 高频纹理和 RGB 低频颜色，导致细节模糊、颜色失真。
                    ②频域建模不足：缺乏长距离依赖建模，无法捕捉全局关联，NIR 高频细节易被抑制。②特征细化需求：融合后易留噪声，需多尺度精细化处理。
            实现方式：①公共特征强化（CRM）：计算频域长距离相关性与局部点相关性，强化共有结构。
                    ②差异特征强化（DRM）：用差异交叉注意力提取NIR高频纹理（RGB缺失部分），可学习参数λ动态调整特征权重。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

# 将输入张量从4D(b,c,h,w)转为3D(b,h*w,c)格式
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

# 将输入张量从3D(b,h*w,c)转回4D(b,c,h,w)格式
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# 无偏置的LayerNorm实现
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        # 处理不同格式的normalized_shape输入
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # 确保输入维度为1
        assert len(normalized_shape) == 1

        # 可学习的权重参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 计算方差并进行归一化
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

# 带偏置的LayerNorm实现
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        # 处理不同格式的normalized_shape输入
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # 确保输入维度为1
        assert len(normalized_shape) == 1

        # 可学习的权重和偏置参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 计算均值和方差并进行归一化
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

# 统一的LayerNorm接口，支持有偏置和无偏置两种模式
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        # 根据类型选择对应的LayerNorm实现
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # 保存原始的高度和宽度信息
        h, w = x.shape[-2:]
        # 先转为3D进行归一化，再转回4D
        return to_4d(self.body(to_3d(x)), h, w)

# 计算注意力机制中的lambda初始值，基于网络深度的衰减函数
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

# 特征增强融合模块(FEFM)，结合频域和空域信息处理多模态特征
class FEFM(nn.Module):
    def __init__(self, dim, bias, depth):
        super(FEFM, self).__init__()
        # 计算注意力头的数量
        self.num_heads = dim // 16
        # 初始化lambda参数
        self.lambda_init = lambda_init_fn(depth)
        # 可学习的lambda参数，用于控制注意力机制
        self.lambda_q1 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        # 各种卷积层，用于特征提取和转换
        self.to_hidden = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.to_hidden_nir = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.to_hidden_dw_nir = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2,
                                          bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.special = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # 注意力温度参数，控制注意力分布的锐度
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

        # 输出投影层
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_middle = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 特征池化层，用于特征融合
        self.pool1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False))
        self.pool2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False))

        # 层归一化
        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')

        # 分块大小，用于分块处理
        self.patch_size = 8

    def forward(self, x, nir):
        # 获取输入特征的维度信息
        b, c, h, w = x.shape

        # 提取RGB和NIR特征 【Conv】
        hidden = self.to_hidden(x)
        nir_hidden = self.to_hidden_nir(nir)
        # 生成查询(Q)、键(K)和值(V)
        q = self.to_hidden_dw(hidden)
        k, v = self.to_hidden_dw_nir(nir_hidden).chunk(2, dim=1)

        # 将特征图分块处理
        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,patch2=self.patch_size)

        # 应用二维离散余弦变换(DCT)转换到频域
        q_fft = dct.dct_2d(q_patch.float())
        k_fft = dct.dct_2d(k_patch.float())

        # 频域特征交互
        out1 = q_fft * k_fft

        # 重塑张量，准备进行多头注意力计算
        q_fft = rearrange(q_fft, 'b (head c) h w patch1 patch2-> b head c (h w patch1 patch2)', head=self.num_heads)
        k_fft = rearrange(k_fft, 'b (head c) h w patch1 patch2-> b head c (h w patch1 patch2)', head=self.num_heads)

        # 【CRM】
        out1 = rearrange(out1, 'b (head c) h w patch1 patch2-> b head c (h w patch1 patch2)', head=self.num_heads)
        # 归一化处理
        q_fft = torch.nn.functional.normalize(q_fft, dim=-1)
        k_fft = torch.nn.functional.normalize(k_fft, dim=-1)
        # 计算注意力矩阵
        attn = (q_fft @ k_fft.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # 应用注意力权重
        out = (attn @ out1)
        # 重塑张量回原始分块格式
        out = rearrange(out, 'b head c (h w patch1 patch2) -> b (head c) h w patch1 patch2', head=self.num_heads,
                        h=h // self.patch_size, w=w // self.patch_size, patch1=self.patch_size, patch2=self.patch_size)

        # 应用二维逆离散余弦变换(IDCT)转回空域
        out = dct.idct_2d(out)

        # 重塑张量回原始特征图格式
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,patch2=self.patch_size)

        # 中间投影
        out = self.project_middle(out)

        # 计算注意力控制参数
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 特征融合
        output = self.pool1(q * out) + self.pool2(v - lambda_full * (v * out))

        # 最终投影输出
        output = self.project_out(output)
        return output

if __name__ == '__main__':
    block = FEFM(dim=32, bias=False, depth=1)
    x0 = torch.randn((1, 32, 64, 64))
    x1 = torch.randn((1, 32, 64, 64))
    output = block(x0, x1)
    print(f"输入张量形状: {x0.shape}")
    print(f"输入张量形状: {x1.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")