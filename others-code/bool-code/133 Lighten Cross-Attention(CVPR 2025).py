import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

"""
    论文地址：https://arxiv.org/abs/2502.20272
    论文题目：HVI: A New Color Space for Low-light Image Enhancement（CVPR 2025）
    中文题目：HVI：一种用于低光图像增强的新颜色空间 （CVPR 2025）
    讲解视频：https://www.bilibili.com/video/BV1AGXzYXEhC/
    轻量化交叉注意力机制（Lighten Cross-Attention，LCA）：
        实际意义：①抑制低光噪声：低光图像噪声多，影响质量。LCA 模块中的 CDL 基于物理光学原理，去除颜色噪声，减少颜色偏差，让图像更清晰。
        ②优化强度增强：低光图像亮度不足，IEL 基于 Retinex 理论优化强度合理调整亮度，避免过暗或局部过亮，保持图像结构和纹理，让图像亮度均匀、细节清晰。
        实现方式：以代码为准。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        # 可学习的权重参数，初始化为全 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # 可学习的偏置参数，初始化为全 0
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        # 用于数值稳定性的小常数
        self.eps = eps
        # 数据格式，支持 "channels_last" 和 "channels_first"
        self.data_format = data_format
        # 检查数据格式是否合法，若不合法则抛出异常
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        # 归一化的形状
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # 如果数据格式为 "channels_last"
        if self.data_format == "channels_last":
            # 直接调用 PyTorch 的层归一化函数
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 如果数据格式为 "channels_first"
        elif self.data_format == "channels_first":
            # 计算通道维度上的均值
            u = x.mean(1, keepdim=True)
            # 计算通道维度上的方差
            s = (x - u).pow(2).mean(1, keepdim=True)
            # 进行归一化操作
            x = (x - u) / torch.sqrt(s + self.eps)
            # 应用可学习的权重和偏置
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# Intensity Enhancement Layer，强度增强层
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        # 调用父类的构造函数
        super(IEL, self).__init__()
        # 计算隐藏层的特征维度
        hidden_features = int(dim * ffn_expansion_factor)
        # 输入投影层，将输入特征维度映射到隐藏层特征维度的 2 倍
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 深度可分离卷积层 1
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        # 深度可分离卷积层 2
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        # 深度可分离卷积层 3
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        # 输出投影层，将隐藏层特征维度映射回输入特征维度
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        # Tanh 激活函数
        self.Tanh = nn.Tanh()

    def forward(self, x):
        # 输入投影
        x = self.project_in(x)
        # 将特征图在通道维度上拆分为两部分
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # 对 x1 应用深度可分离卷积和 Tanh 激活函数，并加上残差连接
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        # 对 x2 应用深度可分离卷积和 Tanh 激活函数，并加上残差连接
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        # 将 x1 和 x2 逐元素相乘
        x = x1 * x2
        # 输出投影
        x = self.project_out(x)
        return x

# Cross Attention Block，交叉注意力块
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        # 调用父类的构造函数
        super(CAB, self).__init__()
        # 注意力头的数量
        self.num_heads = num_heads
        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # 查询卷积层
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # 查询的深度可分离卷积层
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # 键值卷积层
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        # 键值的深度可分离卷积层
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        # 输出投影层
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # 获取输入特征图的形状
        b, c, h, w = x.shape
        # 计算查询
        q = self.q_dwconv(self.q(x))

        # 计算键值
        kv = self.kv_dwconv(self.kv(y))
        # 将键值在通道维度上拆分为键和值
        k, v = kv.chunk(2, dim=1)

        # 对查询、键和值进行维度重排
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 对查询和键进行归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # 对注意力分数进行 softmax 操作
        attn = nn.functional.softmax(attn, dim=-1)
        # 计算注意力输出
        out = (attn @ v)
        # 对注意力输出进行维度重排
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # 输出投影
        out = self.project_out(out)
        return out

# Lightweight Cross Attention，轻量级交叉注意力
class LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        # 调用父类的构造函数
        super(LCA, self).__init__()
        # 强度增强层
        self.gdfn = IEL(dim)
        # 层归一化层
        self.norm = LayerNorm(dim)
        # 交叉注意力块
        self.ffn = CAB(dim, num_heads, bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x)) + x
        return x

if __name__ == "__main__":
    # 创建 LCA 模块实例
    module =  LCA(dim=64,num_heads=8)
    # 生成随机输入张量 x
    input_x = torch.randn(1, 64, 32, 32)
    # 生成随机输入张量 y
    input_y = torch.randn(1, 64, 32, 32)
    # 计算输出张量
    output_tensor = module(input_x,input_y)
    # 打印输入张量的形状
    print('Input size:', input_x.size())
    # 打印输出张量的形状
    print('Output size:', output_tensor.size())
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")