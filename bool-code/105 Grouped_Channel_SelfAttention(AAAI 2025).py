# 导入必要的库
import torch.nn.functional as F  # PyTorch神经网络函数库
from einops import rearrange  # 张量维度操作库
import numbers  # 数字类型判断库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块

"""
    论文地址：https://arxiv.org/abs/2404.07846
    论文题目：Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising（AAAI 2025）
    中文题目：基于Transformer的自监督图像去噪盲点网络的再思考(AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV1LPP9eyE9J/
    分组通道自注意力（Grouped Channel-Wise Self-Attention，G-CSA）：
        发现问题： 空间信息主要通过全局平均池化进行压缩，使得通道注意力（CA）能够在通道维度上有效地捕捉不同通道之间的依赖关系，从而重新校准通道特征响应，提升模型的去噪性能。
                 在多尺度特征中，空间信息会通过下采样操作被打乱并分配到不同的通道中。此时，由于CA聚合所有空间位置的内容，就容易泄漏像素的真实值信息，这会导致模型过拟合到噪声输入，进而影响去噪效果。【希望他学习，但是这里他直接抄】
        理论支撑：为解决多尺度架构中通道注意力（CA）泄漏盲点信息的问题，提出控制通道数小于空间分辨率，将深层特征划分为多个通道组，然后分别对每个通道组执行 CA 操作。
        实现方式：其实就是多头自注意力（代码部分可以看出）
"""
# 张量维度转换函数 ============================================================
def to_3d(x):
    """将四维张量[B, C, H, W]转换为三维[B, H*W, C]格式（用于序列处理）"""
    # 使用einops库的rearrange函数重组维度
    # b: batch_size, c: channels, h: height, w: width
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    """将三维张量[B, H*W, C]恢复为四维[B, C, H, W]格式（恢复空间结构）"""
    # 逆向重组维度操作
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# 层归一化模块 ==============================================================
class BiasFree_LayerNorm(nn.Module):
    """无偏置的层归一化实现（仅含可学习缩放参数）"""

    def __init__(self, normalized_shape):
        super().__init__()
        # 标准化输入形状处理（确保是元组形式）
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1  # 只支持对最后一维归一化

        # 可学习参数（缩放因子）
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """前向传播过程"""
        # 计算方差（沿最后一个维度，保持维度用于广播）
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 归一化公式：x / sqrt(方差 + eps) * 缩放因子
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    """带偏置的层归一化实现（包含缩放和偏移参数）"""
    def __init__(self, normalized_shape):
        super().__init__()
        # 参数初始化与BiasFree版本类似
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        # 可学习参数（缩放因子 + 偏移量）
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """前向传播过程"""
        # 计算均值（沿最后一个维度）
        mu = x.mean(-1, keepdim=True)
        # 计算方差
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 归一化公式：(x - 均值)/sqrt(方差 + eps)*缩放 + 偏移
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    """动态选择归一化类型的层归一化模块"""

    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        # 根据类型选择具体实现
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        """前向传播包含维度转换步骤"""
        # 获取原始空间维度
        h, w = x.shape[-2:]
        # 转换为3D -> 归一化 -> 恢复为4D
        return to_4d(self.body(to_3d(x)), h, w)

# 核心注意力模块 ============================================================
# Grouped_Channel_SelfAttention
class Grouped_Channel_SelfAttention(nn.Module):
    """带空洞的多头通道注意力机制"""
    def __init__(self, dim, num_heads=2, bias=False):
        super().__init__()
        self.num_heads = num_heads  # 注意力头数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 可学习温度系数

        # 1x1卷积生成QKV
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 3x3空洞深度卷积（扩张率2）
        """
            扩张卷积能够在不增加参数数量的情况下扩大感受野，并且通过合适的填充（这里是padding = 2），可以避免卷积操作跨越盲点区域，从而满足盲点网络的要求，防止空间信息泄漏。
        """
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1,
            dilation=2, padding=2, groups=dim * 3, bias=bias
        )
        # 输出投影层
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # 归一化层（使用无偏置版本）
        self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

    def forward(self, x):
        """前向传播过程"""
        # 归一化输入
        x = self.norm(x)
        b, c, h, w = x.shape

        # 生成QKV特征
        qkv = self.qkv_dwconv(self.qkv(x))  # [B, C, H, W] === >[B, 3C, H, W]

        # 分割为Q、K、V三部分
        q, k, v = qkv.chunk(3, dim=1)  # 各为[B, C, H, W]

        # 重组为多头格式
        """
            文中强调：设置每组的通道数足够小，以避免空间信息泄漏。
            实际实现：1、代码中没有直接显式地控制每组通道数与空间分辨率的关系，但[多头机制在一定程度上起到了类似的作用]。
                    2、通过将特征划分为多个头（通道组），每个头处理的通道数相对减少，有助于减少空间信息泄漏的风险。
                    3、每个头可以看作一个通道组，分别在这些通道组上进行后续的注意力计算，与文章中分组执行 CA 的思想一致。
        """
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化Q和K（稳定训练）
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 计算注意力分数（缩放点积注意力）
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)  # 注意力权重归一化

        # 应用注意力到V
        out = attn @ v  # [B, head, C, H*W]

        # 重组输出维度
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=h, w=w)
        # 最终投影
        return self.project_out(out)

# 模块测试 =================================================================
if __name__ == "__main__":
    # 实例化注意力模块（输入通道64）
    DilatedMDTA_channel_attn = Grouped_Channel_SelfAttention(64)
    # 创建测试输入（1个样本，64通道，128x128分辨率）
    input_tensor = torch.randn(1, 64, 128, 128)
    # 前向传播
    output_tensor = DilatedMDTA_channel_attn(input_tensor)
    # 打印输入输出尺寸
    print('输入尺寸:', input_tensor.size())  # 应输出 torch.Size([1, 64, 128, 128])
    print('输出尺寸:', output_tensor.size())  # 应保持相同尺寸
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")