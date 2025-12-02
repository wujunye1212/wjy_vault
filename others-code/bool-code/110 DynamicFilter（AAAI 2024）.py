import torch
import torch.nn as nn
from timm.layers.helpers import to_2tuple
"""
    论文地址：https://ojs.aaai.org/index.php/AAAI/article/download/29457/30746
    论文题目：FFT-Based Dynamic Token Mixer for Vision（AAAI 2024）
    中文题目：基于快速傅里叶变换（FFT）的视觉动态令牌混合器
    讲解视频：https://www.bilibili.com/video/BV1wxNdewEVA/
        基于傅里叶变换的动态滤波器（Dynamic Filter)：
        解决问题：Transformer在计算机视觉多任务中广泛应用，其核心的MHSA模块虽有优势，但因全局注意力设计导致计算复杂度与像素数平方成正比，处理高分辨率图像时速度慢。
        实现方式：动态滤波器基于一个维度为N的全局滤波器，与传统固定的全局滤波器不同，为每个通道使用线性耦合全局滤波器，滤波器权重系数由一个多层感知器（MLP）来调控。
                这种设计使得滤波器能依据输入数据的特征动态调整，增强模型对不同图像内容适应性。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""
class StarReLU(nn.Module):
    """
    StarReLU激活函数：s * relu(x) ** 2 + b
    其中s和b是可学习的参数。
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()  # 调用父类nn.Module的构造函数
        self.inplace = inplace  # 是否进行原地操作
        self.relu = nn.ReLU(inplace=inplace)  # 定义ReLU激活层
        # 定义可学习的缩放参数s，默认值为scale_value，是否需要梯度更新由scale_learnable决定
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                              requires_grad=scale_learnable)
        # 定义可学习的偏置参数b，默认值为bias_value，是否需要梯度更新由bias_learnable决定
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                             requires_grad=bias_learnable)

    def forward(self, x):
        """
        前向传播函数，计算StarReLU激活后的输出。
        """
        return self.scale * self.relu(x) ** 2 + self.bias  # 应用StarReLU公式

# 定义多层感知机（MLP）类，常用于MetaFormer系列模型
class Mlp(nn.Module):
    """
    MLP模块，类似于Transformer、MLP-Mixer等模型中使用的多层感知机。
    大部分代码来源于timm库。
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()  # 调用父类nn.Module的构造函数
        in_features = dim  # 输入特征维度
        out_features = out_features or in_features  # 输出特征维度，默认与输入相同
        hidden_features = int(mlp_ratio * in_features)  # 隐藏层特征维度
        drop_probs = to_2tuple(drop)  # 将dropout概率转换为二元组

        # 第一个全连接层，将输入维度映射到隐藏层维度
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()  # 激活函数层，使用StarReLU
        self.drop1 = nn.Dropout(drop_probs[0])  # 第一个dropout层
        # 第二个全连接层，将隐藏层维度映射回输出维度
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])  # 第二个dropout层

    def forward(self, x):
        """
        前向传播函数，依次通过全连接层、激活函数和dropout层。
        """
        x = self.fc1(x)  # 第一个全连接层
        x = self.act(x)  # 激活函数
        x = self.drop1(x)  # 第一个dropout层
        x = self.fc2(x)  # 第二个全连接层
        x = self.drop2(x)  # 第二个dropout层
        return x  # 返回输出

# 定义动态滤波器模块，用于处理多维数据
class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14,
                 **kwargs):
        super().__init__()  # 调用父类nn.Module的构造函数
        size = to_2tuple(size)  # 将size转换为二元组（高度，宽度）
        self.size = size[0]  # 高度
        self.filter_size = size[1] // 2 + 1  # 滤波器大小，根据FFT要求调整
        self.num_filters = num_filters  # 滤波器数量
        self.med_channels = int(expansion_ratio * dim)  # 中间通道数

        # 第一个逐点卷积层，将输入维度扩展到中间通道数
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()  # 第一个激活函数层，使用StarReLU
        # 多层感知机模块，用于生成滤波器的权重
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        # 初始化复数权重参数，形状为（高度，滤波器大小，滤波器数量，2）
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()  # 第二个激活函数层，使用Identity
        # 第二个逐点卷积层，将中间通道数还原到输入维度
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        """
        前向传播函数，实现动态滤波操作。
        """
        # 使用permute方法调整维度顺序，以适应DynamicFilter的输入要求（B, W, H, C）
        x = x.permute(0, 2, 3, 1)
        B, H, W, _ = x.shape  # 获取输入张量的形状（批量大小，高度，宽度，通道数）

        # 计算路由权重，通过对输入在空间维度上取平均后通过MLP和softmax得到
        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters, -1).softmax(dim=1)

        x = self.pwconv1(x)  # 第一个逐点卷积层
        x = self.act1(x)  # 第一个激活函数层
        x = x.to(torch.float32)  # 转换数据类型为float32

        # 对输入进行二维实数FFT，沿高度和宽度维度
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # [图左侧 开始]
        complex_weights = torch.view_as_complex(self.complex_weights)  # 将实数权重转换为复数形式
        routeing = routeing.to(torch.complex64)  # 转换路由权重为复数类型
        # 通过爱因斯坦求和约定计算加权后的滤波器
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        weight = weight.view(-1, self.size, self.filter_size, self.med_channels)  # 调整权重形状
        # [图左侧 结束]

        x = x * weight  # 应用滤波器权重

        # 对加权后的结果进行二维逆实数FFT，恢复到原始空间维度
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)  # 第二个激活函数层
        x = self.pwconv2(x)  # 第二个逐点卷积层

        # 调整输出张量的维度顺序回（B, C, W, H）
        output = x.permute(0, 3, 1, 2)
        return output  # 返回输出

if __name__ == '__main__':
    # 定义一个DynamicFilter模块，输入通道数为32，空间维度为64x64
    block = DynamicFilter(32, size=64)  # size==H,W
    # 创建一个随机输入张量，形状为（批量大小16，通道数32，高度64，宽度64）
    input = torch.rand(1, 32, 64, 64)
    output = block(input)  # 通过DynamicFilter模块处理输入

    print(input.size())  # 打印输入张量的形状
    print(output.size())  # 打印输出张量的形状
    print("公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")