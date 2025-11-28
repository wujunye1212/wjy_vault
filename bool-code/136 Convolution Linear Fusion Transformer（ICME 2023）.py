import torch
import torch.nn as nn
from einops import rearrange

"""
    论文地址：https://arxiv.org/pdf/2303.10321
    论文题目：ABC: Attention with Bilinear Correlation for Infrared Small Target Detection（CCF B）
    中文题目：ABC：用于红外小目标检测的双线性相关注意力机制（CCF B）
    讲解视频：https://www.bilibili.com/video/BV1AHZgYQEtG/
    卷积线性融合Transformer（Convolution Linear Fusion Transformer, CLFT）：
        实际意义：①传统 CNN 模型的局限性：传统基于 CNN 的模型缺乏全局建模能力，只能进行局部特征提取，容易受到噪声干扰。
                ②Transformer 结构的不足：Transformer 结构虽具备优秀的全局特征表征能力，但对于红外小目标这种特征不明显的对象，仅依靠 Transformer 难以取得良好的检测效果。
        实现方式：以代码为准。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""
def conv_relu_bn(in_channel, out_channel, dirate):
    # 返回一个顺序容器，按顺序包含卷积层、批量归一化层和ReLU激活函数
    return nn.Sequential(
        # 二维卷积层，输入通道数为in_channel，输出通道数为out_channel，
        # 卷积核大小为3，步长为1，填充为dirate，膨胀率为dirate
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=dirate,
                  dilation=dirate),
        # 批量归一化层，对out_channel个通道进行归一化
        nn.BatchNorm2d(out_channel),
        # ReLU激活函数，inplace=True表示直接在输入上进行修改以节省内存
        nn.ReLU(inplace=True)
    )

# 定义BAM（注意力机制）模块
class BAM(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):
        # 调用父类nn.Module的构造函数
        super(BAM, self).__init__()
        # 查询卷积层，将输入通道数为in_dim的特征图转换为1个通道
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        # 键卷积层，将输入通道数为in_dim的特征图转换为1个通道
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)

        # 查询线性层，将输入特征数为in_feature的向量转换为out_feature个特征
        self.query_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        # 键线性层，将输入特征数为in_feature的向量转换为out_feature个特征
        self.key_line = nn.Linear(in_features=in_feature, out_features=out_feature)

        # 卷积层，将1个通道的特征图转换为in_dim个通道
        self.s_conv = nn.Conv2d(in_channels=1, out_channels=in_dim, kernel_size=1)
        # Softmax激活函数，在最后一个维度上进行归一化
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 计算查询向量
        q = rearrange(self.query_line(rearrange(self.query_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b h 1')
        # 计算键向量
        k = rearrange(self.key_line(rearrange(self.key_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b 1 h')
        # 计算注意力图
        att = rearrange(torch.matmul(q, k), 'b h w -> b 1 h w')
        # 对注意力图进行卷积和Softmax操作
        att = self.softmax(self.s_conv(att))
        return att

# 定义普通卷积模块
class Conv(nn.Module):
    def __init__(self, in_dim):
        # 调用父类nn.Module的构造函数
        super(Conv, self).__init__()
        # 定义一个模块列表，包含3个conv_relu_bn模块，输入和输出通道数都为in_dim，膨胀率为1
        self.convs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, 1) for _ in range(3)])

    def forward(self, x):
        # 依次通过模块列表中的每个卷积模块
        for conv in self.convs:
            x = conv(x)
        return x

# 定义膨胀卷积模块
class DConv(nn.Module):
    def __init__(self, in_dim):
        # 调用父类nn.Module的构造函数
        super(DConv, self).__init__()
        # 定义膨胀率列表
        dilation = [2, 4, 2]
        # 定义一个模块列表，包含根据膨胀率列表生成的conv_relu_bn模块
        self.dconvs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, dirate) for dirate in dilation])

    def forward(self, x):
        # 依次通过模块列表中的每个膨胀卷积模块
        for dconv in self.dconvs:
            x = dconv(x)
        return x

# 定义卷积注意力模块
class ConvAttention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):
        # 调用父类nn.Module的构造函数
        super(ConvAttention, self).__init__()
        # 定义普通卷积模块
        self.conv = Conv(in_dim)
        # 定义膨胀卷积模块
        self.dconv = DConv(in_dim)
        # 定义注意力模块
        self.att = BAM(in_dim, in_feature, out_feature)
        # 定义可学习的参数gamma，初始值为0
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 通过普通卷积模块
        q = self.conv(x)
        # 通过膨胀卷积模块
        k = self.dconv(x)
        # 将普通卷积和膨胀卷积的输出相加
        v = q + k
        # 计算注意力图 【图中 BAM】
        att = self.att(x)
        # 计算注意力加权后的输出
        out = torch.matmul(att, v)
        # 返回最终输出，结合注意力输出、v和输入x
        return self.gamma * out + v + x

# 定义前馈网络模块
class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim):
        # 调用父类nn.Module的构造函数
        super(FeedForward, self).__init__()
        # 定义卷积、ReLU激活和批量归一化的组合模块
        self.conv = conv_relu_bn(in_dim, out_dim, 1)
        # 定义一个卷积层，将输入通道数为in_dim的特征图转换为out_dim个通道
        # self.x_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        # 定义一个顺序容器，包含卷积层、批量归一化层和ReLU激活函数
        self.x_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 通过卷积模块
        out = self.conv(x)
        # 通过x_conv模块
        x = self.x_conv(x)
        # 将两个输出相加
        return x + out

# 定义CLFT模块
class CLFT(nn.Module):
    def __init__(self, in_dim, out_dim, in_feature, out_feature):
        # 调用父类nn.Module的构造函数
        super(CLFT, self).__init__()
        # 定义卷积注意力模块
        self.attention = ConvAttention(in_dim, in_feature, out_feature)
        # 定义前馈网络模块
        self.feedforward = FeedForward(in_dim, out_dim)

    def forward(self, x):
        # 通过卷积注意力模块
        x = self.attention(x)
        # 通过前馈网络模块
        out = self.feedforward(x)
        return out

if __name__ == '__main__':
    # 创建CLFT模型实例
    model = CLFT(64, 64, 32 * 32, 32)
    # 生成随机输入张量
    input_tensor = torch.randn(1, 64, 32, 32)
    # 打印输入张量的形状
    print("Input shape:", input_tensor.shape)
    # 通过模型得到输出张量
    output_tensor = model(input_tensor)
    # 打印输出张量的形状
    print("Output shape:", output_tensor.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")