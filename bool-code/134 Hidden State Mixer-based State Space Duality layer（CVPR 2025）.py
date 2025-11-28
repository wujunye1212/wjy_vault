import torch
import torch.nn as nn
import math
"""
    论文地址：https://arxiv.org/pdf/2411.15241
    论文题目：EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality（CVPR 2025）
    中文题目：高效ViM：基于隐藏状态混合器的状态空间对偶性的高效视觉Mamba（CVPR 2025）
    讲解视频：https://www.bilibili.com/video/BV1xUoCYEEyJ/
    基于隐藏状态混合器的状态空间对偶层（HSM-SSD）：
        实际意义：①降低计算成本：传统的基于注意力机制的模型在捕捉全局依赖时，自注意力的二次计算复杂度限制了其效率和可扩展性。
                ②提升模型性能：引入多阶段隐藏状态融合，整合高低层特征，增强泛化能力。
        实现方式：以代码为准。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""
class LayerNorm1D(nn.Module):
    """LayerNorm for channels of 1D tensor(B C L)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        # 调用父类的构造函数
        super(LayerNorm1D, self).__init__()
        # 通道数量
        self.num_channels = num_channels
        # 防止除零的小常数
        self.eps = eps
        # 是否使用可学习的仿射变换
        self.affine = affine

        if self.affine:
            # 可学习的权重参数，形状为 (1, num_channels, 1)
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            # 可学习的偏置参数，形状为 (1, num_channels, 1)
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            # 不使用可学习的权重参数
            self.register_parameter('weight', None)
            # 不使用可学习的偏置参数
            self.register_parameter('bias', None)

    def forward(self, x):
        # 计算每个样本在通道维度上的均值，保持维度不变
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        # 计算每个样本在通道维度上的方差，保持维度不变，不使用无偏估计
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        # 对输入进行归一化处理
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            # 应用可学习的仿射变换
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm2d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        # 调用父类的构造函数
        super(ConvLayer2D, self).__init__()
        # 定义二维卷积层
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
        # 如果指定了归一化层，则创建归一化层，否则为 None
        self.norm = norm(num_features=out_dim) if norm else None
        # 如果指定了激活层，则创建激活层，否则为 None
        self.act = act_layer() if act_layer else None

        if self.norm:
            # 初始化归一化层的权重
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            # 初始化归一化层的偏置
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通过卷积层
        x = self.conv(x)
        if self.norm:
            # 通过归一化层
            x = self.norm(x)
        if self.act:
            # 通过激活层
            x = self.act(x)
        return x

class ConvLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm1d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        # 调用父类的构造函数
        super(ConvLayer1D, self).__init__()
        # 定义一维卷积层
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
        # 如果指定了归一化层，则创建归一化层，否则为 None
        self.norm = norm(num_features=out_dim) if norm else None
        # 如果指定了激活层，则创建激活层，否则为 None
        self.act = act_layer() if act_layer else None

        if self.norm:
            # 初始化归一化层的权重
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            # 初始化归一化层的偏置
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通过卷积层
        x = self.conv(x)
        if self.norm:
            # 通过归一化层
            x = self.norm(x)
        if self.act:
            # 通过激活层
            x = self.act(x)
        return x

class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim=64):
        # 调用父类的构造函数
        super().__init__()
        # 扩展因子
        self.ssd_expand = ssd_expand
        # 扩展后的维度
        self.d_inner = int(self.ssd_expand * d_model)
        # 状态维度
        self.state_dim = state_dim

        # 定义一维卷积层，用于投影
        self.BCdt_proj = ConvLayer1D(d_model, 3 * state_dim, 1, norm=None, act_layer=None)
        # 卷积层的输入维度
        conv_dim = self.state_dim * 3
        # 定义二维深度可分离卷积层
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0)
        # 定义一维卷积层，用于投影
        self.hz_proj = ConvLayer1D(d_model, 2 * self.d_inner, 1, norm=None, act_layer=None)
        # 定义一维卷积层，用于输出投影
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        # 初始化 A 参数
        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        # 将 A 转换为可学习的参数
        self.A = torch.nn.Parameter(A)
        # 定义激活函数
        self.act = nn.SiLU()
        # 定义可学习的参数 D
        self.D = nn.Parameter(torch.ones(1))
        # 标记 D 不进行权重衰减
        self.D._no_weight_decay = True
        # 定义归一化层
        self.norm = LayerNorm1D(d_model)

    def forward(self, x):
        # 对输入进行归一化处理，并将最后两个维度展平
        x = self.norm(x.flatten(2))

        # 获取批量大小、通道数和序列长度
        batch, _, L = x.shape
        # 计算 H，假设 L 是一个完全平方数
        H = int(math.sqrt(L))

        # 通过 BCdt_proj 卷积层，然后调整形状，再通过 dw 卷积层，最后展平
        BCdt = self.dw(self.BCdt_proj(x).view(batch, -1, H, H)).flatten(2)

        # 将 BCdt 拆分为 B、C、dt
        B, C, dt = torch.split(BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)
        # 计算 A
        A = (dt + self.A.view(1, -1, 1)).softmax(-1)

        # 计算 AB
        AB = (A * B)
        # 计算 h
        h = x @ AB.transpose(-2, -1)

        # 将 hz_proj 的输出拆分为 h 和 z
        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        # 计算 h
        h = self.out_proj(h * self.act(z) + h * self.D)

        # 计算 y
        y = h @ C  # B C N, B C L -> B C L
        # 调整 y 的形状
        y = y.view(batch, -1, H, H).contiguous()  # + x * self.D  # B C H W
        return y


if __name__ == '__main__':
    block = HSMSSD(d_model=64)
    input_tensor = torch.rand(1, 64, 32, 32)
    output = block(input_tensor)

    print("Input size:", input_tensor.size())
    print("Output size:", output.size())
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")