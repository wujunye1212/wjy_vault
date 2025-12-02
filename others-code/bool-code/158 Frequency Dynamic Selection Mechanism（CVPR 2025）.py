import torch
import torch.nn as nn
import numbers
from einops import rearrange  # 用于灵活调整张量维度的工具库
from timm.layers import to_2tuple  # 来自timm库的工具函数，用于确保参数是元组形式

"""    
    论文地址：https://arxiv.org/pdf/2412.16645v1
    论文题目：Complementary Advantages: Exploiting Cross-Field Frequency Correlation for NIR-Assisted Image Denoising （CVPR 2025）
    中文题目：互补优势：利用跨域频率相关性实现近红外辅助图像去噪 （CVPR 2025）
    讲解视频：https://www.bilibili.com/video/BV1Z2jyz2EHK/
        频率动态选择机制（Frequency Dynamic Selection Mechanism，FDSM）：
            实际意义：①无效特征混合：不同模态的颜色、亮度、结构差异大，传统固定卷积无法灵活提取互补特征。
                    ②互补信息丢失：RGB低频保色、高频含噪；NIR低频信息少、高频纹理清晰，两个模态之间相关的高频纹理细节与低频颜色信息难以被有效分离和保留。
            实现方式：①动态滤波生成：池化→MLP→Softmax→频域滤波器。
                   ②特征筛选：傅里叶变换→频域点乘→逆变换→输出有效特征
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

def to_3d(x):
    """将4维张量转换为3维张量（空间维度展平）"""
    # 输入形状: [batch, channel, height, width] -> 输出形状: [batch, height*width, channel]
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """将3维张量恢复为4维张量（还原空间维度）"""
    # 输入形状: [batch, height*width, channel] -> 输出形状: [batch, channel, height, width]
    # h: 目标高度, w: 目标宽度
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    """无偏置的层归一化（LayerNorm）实现"""

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        # 处理输入形状参数（可以是整数或元组）
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)

        # 确保归一化维度是1维（通常是特征维度）
        assert len(self.normalized_shape) == 1

        # 可学习的权重参数（初始化为1）
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        """前向传播"""
        # 计算最后一维的方差（保持维度以便广播）
        sigma = x.var(-1, keepdim=True, unbiased=False)  # unbiased=False使用有偏方差估计
        # 归一化公式：x / sqrt(方差 + 小量) * 权重
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """带偏置的层归一化实现（标准LayerNorm）"""

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        # 处理输入形状参数（同BiasFree_LayerNorm）
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)

        assert len(self.normalized_shape) == 1

        # 可学习的权重和偏置参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        """前向传播"""
        # 计算最后一维的均值（保持维度）
        mu = x.mean(-1, keepdim=True)
        # 计算最后一维的方差（保持维度）
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 标准归一化公式：(x - 均值) / sqrt(方差 + 小量) * 权重 + 偏置
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """层归一化封装类（可选择是否带偏置）"""

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        # 根据类型选择归一化实现
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)  # 无偏置版本
        else:
            self.body = WithBias_LayerNorm(dim)  # 带偏置版本

    def forward(self, x):
        """前向传播（自动处理维度转换）"""
        # 获取输入的高和宽（用于恢复4维形状）
        h, w = x.shape[-2:]
        # 先展平为3维输入归一化层，再恢复为4维输出
        return to_4d(self.body(to_3d(x)), h, w)


class StarReLU(nn.Module):
    """自定义激活函数StarReLU：s * relu(x)² + b"""

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace  # 是否原地操作（节省内存）
        self.relu = nn.ReLU(inplace=inplace)  # 基础ReLU激活
        # 可学习的缩放参数（初始化为scale_value）
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        # 可学习的偏置参数（初始化为bias_value）
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        """前向传播"""
        return self.scale * (self.relu(x) ** 2) + self.bias  # 公式实现


def resize_complex_weight(origin_weight, new_h, new_w):
    """调整复数权重的尺寸（用于动态滤波器适配不同输入尺寸）"""
    # 原始权重形状: [h, w, num_heads, 2]（最后一维是实部+虚部）
    h, w, num_heads = origin_weight.shape[0:3]
    # 调整维度为[1, num_heads*2, h, w]（适配插值输入要求）
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
    # 双三次插值调整尺寸到[new_h, new_w]
    new_weight = torch.nn.functional.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    )
    # 恢复原始维度顺序和形状
    new_weight = new_weight.permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)
    return new_weight


class Mlp(nn.Module):
    """多层感知机（常见于Transformer等模型）"""

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim  # 输入特征维度
        out_features = out_features or in_features  # 输出特征维度（默认等于输入）
        hidden_features = int(mlp_ratio * in_features)  # 隐藏层维度（mlp_ratio倍输入维度）
        drop_probs = to_2tuple(drop)  # 确保dropout概率是二元组（对应两个Dropout层）

        # 第一层全连接：输入->隐藏层
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()  # 激活函数（默认使用StarReLU）
        self.drop1 = nn.Dropout(drop_probs[0])  # 第一层Dropout
        # 第二层全连接：隐藏层->输出
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])  # 第二层Dropout

    def forward(self, x):
        x = self.fc1(x)  # 全连接层1
        x = self.act(x)  # 激活函数
        x = self.drop1(x)  # Dropout
        x = self.fc2(x)  # 全连接层2
        x = self.drop2(x)  # Dropout
        return x


class DynamicFilter(nn.Module):
    """动态滤波器模块（核心：频域特征操作）"""

    def __init__(self, dim, expansion_ratio=1, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=30, weight_resize=True,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)  # 确保size是二元组
        self.size = size[0]  # 基础尺寸
        self.filter_size = size[1] // 2 + 1  # 滤波器尺寸（频域半边）
        self.num_filters = num_filters  # 滤波器数量
        self.dim = dim  # 输入特征维度
        self.med_channels = int(expansion_ratio * dim)  # 中间通道数（扩展比例）
        self.weight_resize = weight_resize  # 是否调整权重尺寸

        # 第一层点全连接（特征扩展）
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()  # 激活函数1（默认StarReLU）

        # 路由权重生成MLP（用于动态加权滤波器）
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)

        # 可学习的复数权重（频域滤波器）
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,  # 实部+虚部
                        dtype=torch.float32) * 0.02  # 小初始化（防止梯度爆炸）
        )

        self.act2 = act2_layer()  # 激活函数2（默认无）
        # 第二层点全连接（特征压缩回原维度）
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x, y):
        """前向传播（x是输入特征，y是条件特征）"""
        B, H, W, _ = x.shape  # 获取输入特征的批次、高、宽

        # 计算路由权重（对条件特征y全局平均后输入MLP，生成软注意力权重）
        routeing = self.reweight(y.mean(dim=(1, 2))).view(B, self.num_filters, -1).softmax(dim=1)

        # 特征扩展：点全连接 + 激活
        x = self.pwconv1(x)
        x = self.act1(x)

        # 转换到频域（二维实值快速傅里叶变换）
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')  # 结果是复数张量

        # 调整复数权重尺寸（如果需要适配输入尺寸）
        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1], x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())  # 转换为复数张量
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)  # 不调整尺寸

        # 动态加权滤波器（路由权重与复数滤波器加权融合）
        routeing = routeing.to(torch.complex64)  # 路由权重转为复数
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)  # 爱因斯坦求和实现加权

        # 调整权重维度以匹配特征
        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)

        # 频域特征与滤波器相乘（核心操作）
        x = x * weight

        # 转换回空间域（逆快速傅里叶变换）
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        # 后处理：激活 + 点全连接压缩
        x = self.act2(x)
        x = self.pwconv2(x)

        return x


class FDSM(nn.Module):
    """特征动态融合模块（融合RGB和近红外特征）"""

    def __init__(self, c):
        super().__init__()
        # RGB特征卷积（分组卷积，每组1个通道）
        self.conv_rgb = nn.Conv2d(c, c, 1, 1, 0, groups=c, bias=False)
        # 近红外（NIR）特征卷积（同上）
        self.conv_nir = nn.Conv2d(c, c, 1, 1, 0, groups=c, bias=False)
        # 激活函数（SiLU=Swish）
        self.softmax = nn.SiLU()
        # 特征池化模块（卷积+分组卷积+层归一化+PReLU）
        self.pool = nn.Sequential(
            nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),
            LayerNorm(c, LayerNorm_type='WithBias'),  # 带偏置的层归一化
            nn.PReLU()  # 参数化ReLU
        )
        # 注意力参数生成（输出2倍通道用于a和b）
        self.fc_ab = nn.Sequential(nn.Conv2d(c, c * 2, 1, 1, 0, bias=False))
        # 动态滤波器（分别处理RGB和NIR特征）
        self.dynamic_rgb = DynamicFilter(c)
        self.dynamic_nir = DynamicFilter(c)

    def forward(self, rgb, nir):
        """前向传播（输入RGB和NIR特征）"""
        # 初步特征提取（1x1分组卷积）
        feat_1 = self.conv_rgb(rgb)  # RGB特征
        feat_2 = self.conv_nir(nir)  # NIR特征

        # 特征相加融合
        feat_sum = feat_1 + feat_2

        # 池化模块处理（特征增强）
        # 中间四个层
        s = self.pool(feat_sum)
        z = s  # 中间特征

        # 【傅里叶部分】
        # 生成注意力参数（a和b）
        ab = self.fc_ab(z)  # 输出形状: [B, 2c, H, W]
        B, C, H, W = ab.shape
        ab = ab.view(B, 2, C // 2, H, W)  # 拆分为a和b两个部分
        ab = self.softmax(ab)  # 激活
        a, b = ab[:, 0, ...], ab[:, 1, ...]  # 分离a和b

        # 动态滤波器处理（调整特征维度顺序以适配DynamicFilter输入）
        # 注意：DynamicFilter输入要求是[B, H, W, C]，所以需要permute调整
        feat_1 = self.dynamic_rgb(feat_1.permute(0, 2, 3, 1), a.permute(0, 2, 3, 1))
        feat_2 = self.dynamic_nir(feat_2.permute(0, 2, 3, 1), b.permute(0, 2, 3, 1))

        return feat_1.permute(0, 3, 1, 2), feat_2.permute(0, 3, 1, 2)

if __name__ == '__main__':
    block = FDSM(c=32)  # 创建FDSM模块（输入通道32）
    x0 = torch.randn((1, 32, 50, 50))  # 形状: [B, C, H, W]
    x1 = torch.randn((1, 32, 50, 50))
    output0, output1 = block(x0, x1)
    print(f"输入张量形状: {x0.shape}")
    print(f"输入张量形状: {x1.shape}")
    print(f"输出张量形状: {output0.shape}")
    print(f"输出张量形状: {output1.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")