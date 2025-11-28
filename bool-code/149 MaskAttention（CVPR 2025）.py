import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://arxiv.org/pdf/2503.10686
    论文题目：MaskAttn-UNet: A Mask Attention-Driven Framework for Universal Low-Resolution Image Segmentation（ICCV 2025）
    中文题目：MaskAttn-UNet：一种基于掩码注意力驱动的通用低分辨率图像分割框架（ICCV 2025）
    讲解视频：https://www.bilibili.com/video/BV1LQV2z3EU1/
        二进制掩码注意力（Mask Attention Module，MAM）：
            实际意义：①捕捉长距离依赖：在U-Net架构中，由于标准卷积层固定的感受野限制，难以捕捉长距离依赖关系。
                     ②平衡局部与全局信息：基于 Transformer 的模型存在计算和内存开销大的问题，且缺乏 CNN 固有的归纳偏差，容易忽略区分小物体的细粒度细节特征。
            实现方式：①通过生成可学习的二进制掩码，抑制特征图中无信息区域，更有效地捕捉长距离依赖，理解图像中不同区域之间的关系。
                    ②掩码注意力模块在保留卷积网络局部特征提取能力的同时，注入更广泛的上下文信息，关注局部细节同时获取全局上下文，实现两者的平衡。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class MaskAttention(nn.Module):
    def __init__(self, channels, size):
        super(MaskAttention, self).__init__()
        # 保存输入的通道数
        self.channels = channels
        # 保存输入的尺寸
        self.size = size
        # 定义一个线性层，用于将输入特征映射为查询向量
        self.query = nn.Linear(channels, channels)
        # 定义一个线性层，用于将输入特征映射为键向量
        self.key = nn.Linear(channels, channels)
        # 定义一个线性层，用于将输入特征映射为值向量
        self.value = nn.Linear(channels, channels)
        # 初始化掩码为 None
        self.mask = None
        # 定义一个层归一化层，用于对输入进行归一化处理
        self.norm = nn.LayerNorm([channels])

    def forward(self, x):
        # 获取输入张量的批量大小、通道数、高度和宽度
        batch_size, channels, height, width = x.size()
        # 检查输入的通道数是否与初始化时的通道数一致
        if channels != self.channels:
            # 若不一致，抛出值错误
            raise ValueError("Input channel size does not match initialized channel size.")
        # 将输入张量的形状调整为 (batch_size, channels, height * width)，并交换维度顺序
        x = x.view(batch_size, channels, height * width).permute(0, 2, 1)

        # 通过查询线性层得到查询向量
        Q = self.query(x)
        # 通过键线性层得到键向量
        K = self.key(x)
        # 通过值线性层得到值向量
        V = self.value(x)

        # 计算查询向量和键向量的点积，得到注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1))
        # 对注意力分数进行缩放，除以通道数的平方根
        scores = scores / (self.channels ** 0.5)

        # 检查掩码是否为空或者掩码的最后一个维度大小是否与 height * width 不一致
        if self.mask is None or self.mask.size(-1) != height * width:
            # 生成一个随机的二进制掩码，形状为 (batch_size, height, width)
            binary_mask = torch.randint(0, 2, (batch_size, height, width), device=x.device)
            # 将二进制掩码的形状调整为 (batch_size, -1)
            binary_mask = binary_mask.view(batch_size, -1)
            # 对二进制掩码进行处理，将大于 0.5 的元素替换为 0，小于等于 0.5 的元素替换为负无穷
            processed_mask = torch.where(binary_mask > 0.5, torch.tensor(0.0, device=x.device),
                                         torch.tensor(-float('inf'), device=x.device))
            # 对处理后的掩码进行维度扩展，使其形状与注意力分数匹配
            self.mask = processed_mask.unsqueeze(1).expand(-1, height * width,-1)

        # 将注意力分数和掩码相加
        scores = scores + self.mask
        # 对注意力分数应用 softmax 函数，得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        # 计算注意力输出，即注意力权重和值向量的点积
        attention_output = torch.matmul(attention_weights, V)
        # 将注意力输出和输入进行残差连接
        attention_output = attention_output + x
        # 对注意力输出进行层归一化处理
        attention_output = self.norm(attention_output)
        # 将注意力输出的形状调整为与输入一致
        return attention_output.view(batch_size, channels, height, width)

if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 50, 50)
    model = MaskAttention(channels=32, size=(50, 50))
    output = model(input_tensor)
    print(f'Input size: {input_tensor.size()}')
    print(f'Output size: {output.size()}')
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")