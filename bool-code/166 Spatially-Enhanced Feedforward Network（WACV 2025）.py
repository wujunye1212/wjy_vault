import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

"""
    论文地址：https://arxiv.org/abs/2411.06318
    论文题目：SEM-Net: Efficient Pixel Modelling for image inpainting with Spatially Enhanced SSM （WACV 2025）
    中文题目：SEM-Net：基于空间增强型选择性状态空间模型（SSM）的高效像素级图像修复建模方法（WACV 2025）
    讲解视频：https://www.bilibili.com/video/BV18QK7zAEST/
        空间增强前馈网络（Spatially-Enhanced Feedforward Network，SEFN）：
            实际意义：①线性序列处理局限性：传统 SSM（如 Mamba）将2D图像像素展开为1D序列处理，仅能捕获线性序列中的长距离依赖，但破坏了2D空间中像素邻接关系。
                     ②局部空间依赖的缺失：对局部空间结构（如边缘、小区域纹理）的建模能力较弱。
            实现方式：①取特征：两个输入。②提空间（上半部分）。③门控调制：图中Gate。
"""

# 将4D张量 [batch, channels, height, width] 转换为3D张量 [batch, height*width, channels]
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

# 将3D张量 [batch, height*width, channels] 转换回4D张量 [batch, channels, height, width]
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# 无偏置的层归一化 (移除了均值中心化)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        # 确保normalized_shape是元组格式
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # 当前实现只支持1D归一化
        assert len(normalized_shape) == 1

        # 创建可学习的缩放参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 计算方差 (沿最后一个维度)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 归一化: x / sqrt(方差 + eps) * 缩放权重
        return x / torch.sqrt(sigma + 1e-5) * self.weight

# 带偏置的层归一化 (标准实现)
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        # 确保normalized_shape是元组格式
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # 当前实现只支持1D归一化
        assert len(normalized_shape) == 1

        # 创建可学习的缩放参数和偏置参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 计算均值 (沿最后一个维度)
        mu = x.mean(-1, keepdim=True)
        # 计算方差 (沿最后一个维度)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 归一化: (x - 均值) / sqrt(方差 + eps) * 缩放权重 + 偏置
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

# 可选的层归一化封装 (支持两种类型)
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        # 根据类型选择不同的归一化实现
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # 获取输入的空间尺寸
        h, w = x.shape[-2:]
        # 转换到3D -> 归一化 -> 转换回4D
        return to_4d(self.body(to_3d(x)), h, w)

# 空间增强的前馈网络 (Spatially Enhanced Feedforward Network)
# Spatially-Enhanced Feedforward Network
class SEFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=True):
        super(SEFN, self).__init__()

        # 计算中间层特征维度
        hidden_features = int(dim * ffn_expansion_factor)

        # 输入投影层 (1x1卷积)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 空间信息融合层
        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        # 融合后的深度卷积
        self.dwconv_afterfusion = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                            groups=hidden_features, bias=bias)

        # 深度卷积 (分组卷积实现)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        # 输出投影层 (1x1卷积)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # 空间信息处理分支
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 下采样
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'WithBias'),  # 使用带偏置的LN
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'WithBias'),  # 使用带偏置的LN
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)  # 上采样

    def forward(self, x, spatial):
        x = self.project_in(x)

        #### 空间分支处理 ####
        y = self.avg_pool(spatial)  # 下采样
        y = self.conv(y)  # 特征提取
        y = self.upsample(y)  # 上采样恢复尺寸
        #### 结束空间分支 ####

        # 主路径: 深度卷积并分割为两部分
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # 融合主路径特征和空间分支特征
        x1 = self.fusion(torch.cat((x1, y), dim=1))
        # 融合后的深度卷积
        x1 = self.dwconv_afterfusion(x1)

        # 门控机制: GELU激活 + 逐元素相乘
        x = F.gelu(x1) * x2
        # 输出投影
        x = self.project_out(x)
        return x

if __name__ == '__main__':
    # 创建两个随机输入张量
    x1 = torch.randn(2, 32, 50, 50)  # [batch, channels, height, width]
    x2 = torch.randn(2, 32, 50, 50)
    model = SEFN(dim=32)
    output = model(x1, x2)
    print(f"输入张量形状: {x1.shape}")
    print(f"输入张量形状: {x2.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")