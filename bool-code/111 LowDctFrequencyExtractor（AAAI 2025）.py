import torch
import torch.nn as nn
import torch.fft
import math

"""
    论文地址：https://arxiv.org/abs/2412.13753
    论文题目：Mesoscopic Insights: Orchestrating Multi-scale & Hybrid Architecture for Image Manipulation Localization(AAAI 2025)
    中文题目：用于图像篡改定位的多尺度混合架构(AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV1ohK5e6Eb4/
    【低频部分】离散余弦变换模块（Discrete Cosine Transform,DCT)：【类似傅里叶变换】
            实际意义：将时域信号转换到频域的数学变换方法，把信号表示成一系列余弦函数的加权和，这些余弦函数具有不同的频率和幅度。在图像处理领域，DCT 能将图像从空间域转换到频率域，将图像信息分解为不同频率的分量。
            实现方式：通过DCT变换将图像转换到频域后，可以分别对高频和低频成分进行分析。高频成分包含图像的细节信息；低频成分反映图像的整体结构。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

# 定义一个名为 LowDctFrequencyExtractor 的类，继承自 nn.Module，用于提取低 DCT 频率分量
class LowDctFrequencyExtractor(nn.Module):
    # 类的初始化方法，接收一个参数 alpha，默认值为 0.95
    def __init__(self, alpha=0.95):
        # 调用父类 nn.Module 的初始化方法
        super(LowDctFrequencyExtractor, self).__init__()
        # 检查 alpha 的值是否在 (0, 1) 区间内，如果不在则抛出 ValueError 异常
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive)")
        # 将传入的 alpha 值赋值给类的属性 self.alpha
        self.alpha = alpha
        # 初始化用于存储水平方向 DCT 矩阵的变量，初始值为 None
        self.dct_matrix_h = None
        # 初始化用于存储垂直方向 DCT 矩阵的变量，初始值为 None
        self.dct_matrix_w = None

    # 定义一个方法，用于创建 DCT 矩阵，参数 N 表示矩阵的大小
    def create_dct_matrix(self, N):
        # 创建一个从 0 到 N-1 的一维张量，并将其形状调整为 (1, N)
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        # 创建一个从 0 到 N-1 的一维张量，并将其形状调整为 (N, 1)
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        # 根据 DCT 矩阵的计算公式计算矩阵元素的值
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        # 将矩阵的第一行元素设置为 1 / sqrt(N)
        dct_matrix[0, :] = 1 / math.sqrt(N)
        # 返回计算得到的 DCT 矩阵
        return dct_matrix

    # 定义一个方法，用于进行二维离散余弦变换（DCT）
    def dct_2d(self, x):
        # 获取输入张量 x 的倒数第二个维度的大小，即高度 H
        H, W = x.size(-2), x.size(-1)
        # 如果水平方向的 DCT 矩阵还未创建，或者其大小与输入张量的高度不匹配
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            # 调用 create_dct_matrix 方法创建水平方向的 DCT 矩阵，并将其移动到与输入张量相同的设备上
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        # 如果垂直方向的 DCT 矩阵还未创建，或者其大小与输入张量的宽度不匹配
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            # 调用 create_dct_matrix 方法创建垂直方向的 DCT 矩阵，并将其移动到与输入张量相同的设备上
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        # 进行二维 DCT 变换，先将输入张量与垂直方向的 DCT 矩阵的转置相乘，再将结果与水平方向的 DCT 矩阵相乘
        return torch.matmul(self.dct_matrix_h, torch.matmul(x, self.dct_matrix_w.t()))

    # 定义一个方法，用于进行二维逆离散余弦变换（IDCT）
    def idct_2d(self, x):
        # 获取输入张量 x 的倒数第二个维度的大小，即高度 H
        H, W = x.size(-2), x.size(-1)
        # 如果水平方向的 DCT 矩阵还未创建，或者其大小与输入张量的高度不匹配
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            # 调用 create_dct_matrix 方法创建水平方向的 DCT 矩阵，并将其移动到与输入张量相同的设备上
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        # 如果垂直方向的 DCT 矩阵还未创建，或者其大小与输入张量的宽度不匹配
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            # 调用 create_dct_matrix 方法创建垂直方向的 DCT 矩阵，并将其移动到与输入张量相同的设备上
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        # 进行二维 IDCT 变换，先将输入张量与垂直方向的 DCT 矩阵相乘，再将结果与水平方向的 DCT 矩阵的转置相乘
        return torch.matmul(self.dct_matrix_h.t(), torch.matmul(x, self.dct_matrix_w))

    # 定义一个方法，用于进行低通滤波
    """
        在 LowDctFrequencyExtractor 类里，alpha 参数用于低通滤波操作。
            低通滤波的目的是保留图像的低频成分，低频成分通常对应图像的大致结构、轮廓等宏观信息。
        默认值 alpha = 0.95：该值比较大，意味着在低通滤波时，小部分频率成分都会被保留下来。
    """
    def low_pass_filter(self, x, alpha):
        # 获取输入张量 x 的倒数第二个和最后一个维度的大小，即高度 h 和宽度 w
        h, w = x.shape[-2:]
        # 创建一个全为 1 的二维张量，形状为 (h, w)，并将其移动到与输入张量相同的设备上
        mask = torch.ones(h, w, device=x.device)
        # 根据 alpha 值计算需要保留的高度和宽度的像素数量
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        # 将掩码矩阵的右下角部分（高频部分）设置为 0
        mask[-alpha_h:, -alpha_w:] = 0
        # 将输入张量与掩码矩阵逐元素相乘，实现低通滤波
        return x * mask

    # 定义前向传播方法，用于定义模型的计算流程
    def forward(self, x):
        # 对输入张量 x 进行二维 DCT 变换
        xq = self.dct_2d(x)
        # 对 DCT 变换后的结果进行低通滤波，保留低频部分
        xq_high = self.low_pass_filter(xq, self.alpha)
        # 对低通滤波后的结果进行二维 IDCT 变换，将其转换回空间域
        xh = self.idct_2d(xq_high)

        # 获取输出张量 xh 的批次大小 B
        B = xh.shape[0]
        # 将每个批次的输出张量展平为一维向量，然后找到每个批次的最小值
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        # 将每个批次的输出张量展平为一维向量，然后找到每个批次的最大值
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        # 对输出张量进行归一化处理，将其值缩放到 [0, 1] 区间
        xh = (xh - min_vals) / (max_vals - min_vals)
        # 返回归一化后的输出张量
        return xh

if __name__ == '__main__':
    # 随机生成一个形状为 (1, 64, 32, 32) 的输入张量
    input_tensor = torch.rand(1, 64, 32, 32)
    # 创建 LowDctFrequencyExtractor 类的实例
    extractor = LowDctFrequencyExtractor()
    # 调用实例的 forward 方法，对输入张量进行处理
    output = extractor(input_tensor)
    print(input_tensor.size())  # 打印输入张量的形状
    print(output.size())  # 打印输出张量的形状
    print("公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")