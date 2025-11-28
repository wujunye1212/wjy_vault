from einops import rearrange
import torch
import torch.nn as nn


# 通道加权方案 CVPR2024热点！附Transformer和CNN两种
# 讲解视频：https://www.bilibili.com/video/BV1fVxueLErc/


class ChannelAttention_Transformer(nn.Module):
    # 初始化函数，传入参数dim（维度），num_heads（头数），bias（是否使用偏置）
    def __init__(self, dim, num_heads, bias):
        # 调用父类的初始化方法
        super(ChannelAttention_Transformer, self).__init__()
        # 将头数存储为类的属性
        self.num_heads = num_heads
        # 定义一个可学习的温度参数，用于缩放点积注意力
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # 使用1x1卷积层生成查询、键和值，输入输出通道均为dim*3，这里假设QKV三个向量长度相同
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 深度可分离卷积层，对QKV进行进一步处理，保持通道数量不变，但增强了局部特征
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 最终输出投影层，将经过注意力机制处理后的特征图映射回原始维度
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    # 前向传播函数
    def forward(self, x):
        # 获取输入张量x的尺寸信息，b代表batch size，c是通道数，h和w分别是高度和宽度
        b, c, h, w = x.shape
        # 首先通过qkv卷积层然后通过深度可分离卷积层处理输入x，得到qkv
        qkv = self.qkv_dwconv(self.qkv(x))

        # 将qkv沿着通道维分成三部分，分别作为查询q、键k和值v
        q, k, v = qkv.chunk(3, dim=1)

        # 使用rearrange函数调整q的形状，使其符合多头注意力机制的要求
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # 同样地，调整k的形状
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # 对v也执行相同的形状调整
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 对q进行归一化处理，确保其在最后一个维度上的范数为1
        q = torch.nn.functional.normalize(q, dim=-1)
        # 对k同样进行归一化处理
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力权重矩阵，注意要乘以之前定义的温度参数来调节注意力分布
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # 应用softmax函数使得每行成为一个概率分布
        attn = attn.softmax(dim=-1)

        # 根据注意力权重加权求和v，得到最终的输出out
        out = (attn @ v)

        # 将out重新排列回与输入相匹配的形状
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 通过最后的卷积层project_out输出结果
        out = self.project_out(out)
        # 返回处理后的特征图
        return out

class ChannelAttention_CNN(nn.Module):
    """
        CALayer利用全局平均池化来获取每个通道的重要性，
        并且通过一系列卷积层调整这些重要性的权重信息，
        最后，根据计算出的权重对输入特征图进行加权。
        【这种类型的层通常被用来执行通道注意力机制】
    """
    # 定义一个名为CALayer的类，继承自nn.Module
    def __init__(self, channel, reduction=16):
        # 初始化函数，接受两个参数：channel（通道数）和reduction（压缩比，默认为16）
        super(ChannelAttention_CNN, self).__init__()  # 调用父类nn.Module的初始化方法

        # 全局平均池化层: 将特征图转换成单个点
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 创建一个自适应平均池化层，输出尺寸为1x1

        # 特征通道降维再升维 --> 产生通道权重
        self.conv_du = nn.Sequential(
            # 第一个卷积层：将输入通道数减少到原来的1/reduction
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            # ReLU激活函数，使用原地操作以节省内存
            nn.ReLU(inplace=True),
            # 第二个卷积层：恢复通道数到原始数量
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            # Sigmoid激活函数，用于生成每个通道的权重
            nn.Sigmoid()
        )

    def forward(self, x):
        # 前向传播函数，接受一个张量x作为输入
        y = self.avg_pool(x)  # 对输入x进行全局平均池化
        y = self.conv_du(y)   # 通过conv_du网络计算得到每个通道的权重
        return x * y          # 将原始输入x与通道权重y相乘，实现对原始特征图的加权

def count_parameters_in_millions(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1000.0  # 转换为兆

if __name__ == '__main__':
    # 生成随机输入张量
    input = torch.randn(1, 32, 64, 64)

    # 实例化模型对象
    modelV1 = ChannelAttention_Transformer(dim=32,num_heads=8,bias=False)
    modelV2 = ChannelAttention_CNN(channel=32)

    # 执行前向传播
    outputV1 = modelV1(input)
    outputV2 = modelV2(input)

    print('input_size:',input.size())       # input_size: torch.Size([1, 32, 64, 64])

    total_params_millions = count_parameters_in_millions(modelV1)
    print(f"Total number of trainable parameters: {total_params_millions:.2f}K") # 4.97K
    print('outputV1_size:',outputV1.size())  # outputV1_size: torch.Size([1, 32, 64, 64])

    total_params_millions = count_parameters_in_millions(modelV2)
    print(f"Total number of trainable parameters: {total_params_millions:.2f}K") # 0.16K
    print('outputV2_size:', outputV2.size())  # outputV2_size: torch.Size([1, 32, 64, 64])

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")