import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2307.01990
    论文题目：Unsupervised Spectral Demosaicing with Lightweight Spectral Attention Networks
    中文题目：基于轻量级光谱注意力网络的无监督光谱去马赛克
    讲解视频：https://www.bilibili.com/video/BV1mi1nBEE3a/
    轻量级光谱注意力（Lightweight Spectral Attention, LSA）：
        实际意义：①参数量大、计算复杂的问题：不适合部署在存储和计算资源有限的设备上。
                ②无监督框架中失效的问题：在无监督训练框架下，传统模块会失效，导致网络无法稳定收敛伪相关的光谱噪声。
        实现方式：通过“空间矩阵 + 通道向量”双层注意力分解结构，LSA 实现了在参数极度精简的情况下保持光谱特征的选择性增强能力。
"""
# 定义通道池化模块（用于在通道维度上进行信息压缩）
class ChannelPool(nn.Module):
    def forward(self, x):
        # 对输入张量在通道维上求均值，得到单通道特征图
        # 输入形状: [B, C, H, W]
        # 输出形状: [B, 1, H, W]
        return torch.mean(x, 1).unsqueeze(1)

# 定义轻量级光谱注意力模块（Lightweight Spectral Attention, LSA）
class LSA(nn.Module):
    def __init__(self, msfa_size, channel, reduction=16):
        super(LSA, self).__init__()

        # 通道压缩模块，用于将多通道特征压缩成单通道特征
        self.compress = ChannelPool()

        # 降采样模块，将空间信息重排为通道信息（反向的 PixelShuffle）
        self.shuffledown = Shuffle_d(msfa_size)

        # 自适应平均池化，将特征图压缩为 1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 第一个全连接层序列，对光谱特征进行注意力权重计算
        self.fc = nn.Sequential(
            nn.Linear(msfa_size**2, msfa_size**2 // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),                                           # 激活函数
            nn.Linear(msfa_size**2 // reduction, msfa_size**2, bias=False),  # 升维
            nn.Sigmoid()                                                     # 输出注意力权重 [0,1]
        )

        # PixelShuffle 实现上采样（与 shuffledown 相反）
        self.shuffleup = nn.PixelShuffle(msfa_size)

        # 第二套注意力机制，用于通道维度的加权（类似 Squeeze-and-Excitation）
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 保存输入张量，用于通道注意力
        buff_x = x

        # 获取输入张量形状
        B, C, H, W = x.size()

        # 【1、合并-->2、重排-->3、权重相乘】
        # （以重排后的结果作为注意力权重）
        # 将输入张量在批次与通道维合并，为后续空间重排做准备
        x = x.view(B * C, 1, H, W)
        # 使用 Shuffle_d 模块进行降采样（空间→通道）
        sq_x = self.shuffledown(x)
        # 获取降采样后张量的维度信息
        b, c, _, _ = sq_x.size()
        # 对每个特征图进行全局平均池化，得到每个通道的全局统计信息
        y = self.avg_pool(sq_x).view(b, c)
        # 通过全连接层计算光谱维度的注意力权重
        y = self.fc(y).view(b, c, 1, 1)
        # 将注意力权重扩展至与输入特征图相同的空间大小
        y = y.expand_as(sq_x)
        # 使用 PixelShuffle 将通道信息恢复至原始空间分辨率
        ex_y = self.shuffleup(y)

        # 将光谱注意力权重与原始特征逐元素相乘
        out = x * ex_y
        # 还原形状为原输入的 [N, C, H, W]
        out = out.view(B, C, H, W)

        # ===== 通道注意力部分（池化作为注意力权重） =====
        b, c, _, _ = buff_x.size()
        # 全局平均池化：获取每个通道的统计信息
        y = self.avg_pool1(buff_x).view(b, c)
        # 计算通道注意力权重
        y = self.fc1(y).view(b, c, 1, 1)
        # 对特征图进行通道加权
        out = out * y.expand_as(out)

        # 返回融合了光谱与通道注意力的输出
        return out

# 定义空间到通道的重排模块（类似 PixelUnshuffle）
class Shuffle_d(nn.Module):
    def __init__(self, scale=2):
        super(Shuffle_d, self).__init__()
        self.scale = scale

    def forward(self, x):
        # 内部函数：实现空间维度到通道维度的重排
        def _space_to_channel(x, scale):
            b, C, h, w = x.size()
            Cout = C * scale ** 2   # 通道数扩展
            hout = h // scale       # 高度缩小
            wout = w // scale       # 宽度缩小
            # 重排操作：将空间块映射到通道维度
            x = x.contiguous().view(b, C, hout, scale, wout, scale)
            x = x.contiguous().permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, Cout, hout, wout)
            return x
        # 调用内部函数完成转换
        return _space_to_channel(x, self.scale)

if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 64, 64)
    model = LSA(msfa_size=4, channel=32)
    output = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")