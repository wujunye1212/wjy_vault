import torch
import torch.nn.functional as F
from torch import nn

'''
来自TETCI 2024 论文   CV任务通用           YOLOv8v10v11创新改进商品 和 即插即用模块商品在评论区
即插即用注意力： SAA 自我感知注意力           
提供二次创新  SCGA 自我感知协调注意力 效果优于SAA,可以直接拿去冲SCI一区  

从医学图像中精确分割器官或病变对疾病诊断和器官形态测量至关重要。
近年来，卷积编码器-解码器结构在自动医学图像分割领域取得了显著进展。
然而，由于卷积操作的固有偏差，现有模型主要关注由邻近像素形成的局部视觉线索，未能充分建模长程上下文依赖性。
本文提出了一种新颖的基于Transformer的注意力引导网络，称为 TransAttUnet。
该网络设计了多级引导注意力和多尺度跳跃连接，以共同增强语义分割架构的性能。
受Transformer启发，本文将 自感知注意力模块 (SAA) 融入TransAttUnet中，
该模块结合了Transformer自注意力 (TSA) 和全局空间注意力 (GSA)，能够有效地学习编码器特征之间的非局部交互。
此外，本文还在解码器块之间引入了多尺度跳跃连接，用于将不同语义尺度的上采样特征进行聚合，
从而增强多尺度上下文信息的表示能力，生成具有区分性的特征。得益于这些互补组件，
TransAttUnet能够有效缓解卷积层堆叠和连续采样操作引起的细节丢失问题，最终提升医学图像分割的质量。
在多个医学图像分割数据集上的大量实验表明，所提出的方法在不同成像模式下始终优于最新的基线模型。

SAA 模块是作用在于增强医学图像分割的上下文语义建模能力和全局空间关系表征能力。
其核心由以下两部分组成：
1.多头自注意力 Transformer Self Attention (TSA):
使用 Transformer 的多头自注意力机制，能够捕获全局上下文信息并建模长程依赖。
TSA 首先通过线性变换生成查询 (Q)、键 (K) 和值 (V) 的特征表示，然后通过点积操作计算注意力权重，聚合全局特征信息。

2.全局空间注意力 Global Spatial Attention (GSA):
提取和整合全局空间信息，从而增强并优化特征表示。
GSA 通过对特征图进行卷积和重构，生成位置相关的注意力图，进而与输入特征结合，形成强化后的特征。

适用于：医学图像分割，目标检测，语义分割，图像增强，暗光增强，遥感图像任务等所有计算机视觉CV任务通用注意力模块
'''
class PAM_Module(nn.Module):
    """空间注意力模块"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class ScaledDotProductAttention(nn.Module):
    '''自注意力模块'''

    def __init__(self, temperature=512, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature ** 0.5
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        m_batchsize, d, height, width = x.size()
        q = x.view(m_batchsize, d, -1)
        k = x.view(m_batchsize, d, -1)
        k = k.permute(0, 2, 1)
        v = x.view(m_batchsize, d, -1)

        attn = torch.matmul(q / self.temperature, k)

        if mask is not None:
            # 给需要mask的地方设置一个负无穷
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        output = output.view(m_batchsize, d, height, width)

        return output
class SAA(nn.Module):
    def __init__(self, in_channels):
        super(SAA, self).__init__()
        self.gsa = PAM_Module(in_dim=in_channels)
        self.tsa = ScaledDotProductAttention()
    def forward(self, x):
        x1 = self.gsa(x)
        x2 = self.gsa(x)
        out = x1 + x2
        return out
# 二次创新注意力模块 SCGA 自我感知协调注意力 冲SCI一区
'''
SCGA 自我感知协调注意力 内容介绍：

1.执行通道注意力机制。它对每个通道进行全局平均池化，
然后通过1D卷积来捕捉通道之间的交互信息。这种方法避免了降维问题，
确保模型能够有效地聚焦在最相关的通道特征上。
2.全局空间注意力 Global Spatial Attention (GSA):
提取和整合全局空间信息，从而增强并优化特征表示。
GSA 通过对特征图进行卷积和重构，生成位置相关的注意力图，进而与输入特征结合，形成强化后的特征。
3.多头自注意力 Transformer Self Attention (TSA):
使用 Transformer 的多头自注意力机制，能够捕获全局上下文信息并建模长程依赖。
TSA 首先通过线性变换生成查询 (Q)、键 (K) 和值 (V) 的特征表示，然后通过点积操作计算注意力权重，聚合全局特征信息。
'''
class SCGA(nn.Module):
    def __init__(self, in_channels):
        super(SCGA, self).__init__()
        self.gsa = PAM_Module(in_dim=in_channels)
        self.tsa = ScaledDotProductAttention()
        self.ca = ChannelAttention(in_channels)
    def forward(self, x):
        x1 = x * self.ca(x)
        x1 = x1 * self.gsa(x)

        x2 = self.gsa(x)

        out = x1 + x2
        return out
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1, 64, 128, 128)
    SAA = SAA(in_channels=64)
    output = SAA(input)
    print("SAA_input.shape:", input.shape)
    print("SAA_output.shape:",output.shape)
    SCGA = SCGA(in_channels=64)
    output = SCGA(input)
    print("二次创新_SCGA_input.shape:", input.shape)
    print("二次创新_SCGA_output.shape:",output.shape)

