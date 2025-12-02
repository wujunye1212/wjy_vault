from torch import nn
import math
import torch
"""
    论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0925231222003848
    论文题目：FCMNet: Frequency-aware cross-modality attention networks for RGB-D salient object detection
    中文题目：FCMNet：用于 RGB-D 显著物体检测的频率感知跨模态注意力网络
    讲解视频：https://www.bilibili.com/video/BV1ycztYWEzw/
        频率感知跨模态注意力（Frequency-Aware Cross-Modality Attention, FACA）：
           问题：传统注意力机制难以处理跨模态任务中的信息融合问题。
           思路：自动提取和强化不同模态中的互补信息。
           快速梳理：空间频率通道注意(SFCA)从空间和频域中捕获互补信息。RGB分支和深度分支的特征图经过SFCA 模块，然后进行元素乘法以交互不同的模态信息。
"""

# 计算1D DCT系数
def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i+0.5)/L) / math.sqrt(L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)

# 获取DCT权重
def get_dct_weights(width, height, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, height)
    c_part = channel // len(fidx_u)
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                dct_weights[:, i*c_part: (i+1)*c_part, t_x, t_y] = get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height)
    return dct_weights

# 定义FCA模块
class FCABlock(nn.Module):
    """
        FcaNet: Frequency Channel Attention Networks
        https://arxiv.org/pdf/2012.11879.pdf
    """
    def __init__(self, channel, width, height, fidx_u, fidx_v, reduction=16):
        super(FCABlock, self).__init__()
        mid_channel = channel // reduction

        # 预计算DCT权重并注册为缓冲区
        # pre_computed_dct_weights 可以被视为一种不变的超参数。
        # 特点：在模型初始化时被计算并存储，之后在前向传播过程中保持不变。这种设计的好处包括：
        #           1.效率：避免在每次前向传播时重复计算相同的DCT权重。
        #           2. 一致性：确保每次前向传播使用相同的权重，保证模型行为的一致性。
        self.register_buffer('pre_computed_dct_weights', get_dct_weights(width, height, channel, fidx_u, fidx_v))

        # 定义注意力机制的全连接层
        self.excitation = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 计算加权和
        y = torch.sum(x * self.pre_computed_dct_weights, dim=[2, 3])
        # 通过全连接层生成注意力权重
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)

class SFCA(nn.Module):
    def __init__(self, in_channel, width, height, fidx_u, fidx_v):
        super(SFCA, self).__init__()

        # 调整频率索引
        fidx_u = [temp_u * (width // 8) for temp_u in fidx_u]
        fidx_v = [temp_v * (width // 8) for temp_v in fidx_v]
        # 初始化FCA模块
        self.FCA = FCABlock(in_channel, width, height, fidx_u, fidx_v)
        # 定义卷积层和激活函数
        self.conv1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):

        # 计算FCA输出 频域
        F_fca = self.FCA(x)

        # 计算上下文注意力 空间域
        con = self.conv1(x)  # c,h,w -> 1,h,w
        con = self.norm(con)
        F_con = x * con

        return F_fca + F_con

# 定义FACMA模块
class FACMA(nn.Module):
    def __init__(self, in_channel, width, height, fidx_u, fidx_v):
        super(FACMA, self).__init__()
        # 初始化两个SFCA模块
        self.sfca_depth = SFCA(in_channel, width, height, fidx_u, fidx_v)
        self.sfca_rgb = SFCA(in_channel, width, height, fidx_u, fidx_v)

    def forward(self, rgb, depth):
        # 计算深度分支的输出
        out_d = self.sfca_depth(depth)
        out_d = rgb * out_d

        # 计算RGB分支的输出
        out_rgb = self.sfca_rgb(rgb)
        out_rgb = depth * out_rgb
        return out_rgb, out_d

if __name__ == '__main__':
    fidx_u = [0, 1]
    fidx_v = [0, 1]
    # 实例化FACMA
    facma = FACMA(32, 64, 64, fidx_u, fidx_v)

    # 假设的RGB和深度输入
    rgb_input = torch.randn(1, 32, 64, 64)  # Batch size为1
    depth_input = torch.randn(1, 32, 64, 64)  # Batch size为1

    # 通过FACMA
    out_rgb, out_d = facma(rgb_input, depth_input)

    # 打印输入输出形状
    print("RGB图:", rgb_input.shape)
    print("深度图:", depth_input.shape)
    print("RGB图输出:", out_rgb.shape)
    print("深度图输出:", out_d.shape)
    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息