import torch
import torch.nn as nn
import math

"""
    论文地址：https://arxiv.org/abs/2412.13753
    论文题目：Mesoscopic Insights: Orchestrating Multi-scale & Hybrid Architecture for Image Manipulation Localization(AAAI 2025)
    中文题目：用于图像篡改定位的多尺度混合架构(AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV1ohK5e6Eb4/
    【高频部分】离散余弦变换模块（Discrete Cosine Transform,DCT)：【类似傅里叶变换】
            实际意义：将时域信号转换到频域的数学变换方法，把信号表示成一系列余弦函数的加权和，这些余弦函数具有不同的频率和幅度。在图像处理领域，DCT 能将图像从空间域转换到频率域，将图像信息分解为不同频率的分量。
            实现方式：通过DCT变换将图像转换到频域后，可以分别对高频和低频成分进行分析。高频成分包含图像的细节信息；低频成分反映图像的整体结构。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

# 定义一个名为 HighDctFrequencyExtractor 的类，继承自 nn.Module
# 该类的作用是从输入数据中提取高 DCT（离散余弦变换）频率分量
class HighDctFrequencyExtractor(nn.Module):
    # 类的初始化方法，接收一个参数 alpha，默认值为 0.05
    def __init__(self, alpha=0.05):
        # 调用父类 nn.Module 的初始化方法
        super(HighDctFrequencyExtractor, self).__init__()
        # 检查 alpha 的值是否在 (0, 1) 这个开区间内
        # 如果不在这个范围内，抛出一个 ValueError 异常并给出提示信息
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive)")
        # 将传入的 alpha 参数赋值给类的实例属性 self.alpha
        self.alpha = alpha
        # 初始化水平方向的 DCT 矩阵，初始值设为 None
        self.dct_matrix_h = None
        # 初始化垂直方向的 DCT 矩阵，初始值设为 None
        self.dct_matrix_w = None

    # 定义一个方法，用于创建 DCT 矩阵，N 表示矩阵的大小
    def create_dct_matrix(self, N):
        # 创建一个从 0 到 N - 1 的一维张量，并将其形状调整为 (1, N)
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        # 创建一个从 0 到 N - 1 的一维张量，并将其形状调整为 (N, 1)
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        # 根据 DCT 矩阵的计算公式生成矩阵元素
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        # 将矩阵的第一行元素设置为 1 / sqrt(N)
        dct_matrix[0, :] = 1 / math.sqrt(N)
        # 返回生成好的 DCT 矩阵
        return dct_matrix

    # 定义一个方法，用于对输入数据进行二维 DCT 变换
    def dct_2d(self, x):
        # 获取输入张量 x 的倒数第二个维度（高度）和最后一个维度（宽度）的大小
        H, W = x.size(-2), x.size(-1)
        # 检查水平方向的 DCT 矩阵是否未创建，或者其大小与输入数据的高度不匹配
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            # 若不满足条件，则创建一个新的水平方向 DCT 矩阵，并将其放到与输入数据相同的设备上
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        # 检查垂直方向的 DCT 矩阵是否未创建，或者其大小与输入数据的宽度不匹配
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            # 若不满足条件，则创建一个新的垂直方向 DCT 矩阵，并将其放到与输入数据相同的设备上
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        # 进行二维 DCT 变换，先将输入数据与垂直方向 DCT 矩阵的转置相乘，再与水平方向 DCT 矩阵相乘
        return torch.matmul(self.dct_matrix_h, torch.matmul(x, self.dct_matrix_w.t()))

    # 定义一个方法，用于对输入数据进行二维逆 DCT 变换
    def idct_2d(self, x):
        # 获取输入张量 x 的倒数第二个维度（高度）和最后一个维度（宽度）的大小
        H, W = x.size(-2), x.size(-1)
        # 检查水平方向的 DCT 矩阵是否未创建，或者其大小与输入数据的高度不匹配
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            # 若不满足条件，则创建一个新的水平方向 DCT 矩阵，并将其放到与输入数据相同的设备上
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        # 检查垂直方向的 DCT 矩阵是否未创建，或者其大小与输入数据的宽度不匹配
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            # 若不满足条件，则创建一个新的垂直方向 DCT 矩阵，并将其放到与输入数据相同的设备上
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        # 进行二维逆 DCT 变换，先将输入数据与垂直方向 DCT 矩阵相乘，再与水平方向 DCT 矩阵的转置相乘
        return torch.matmul(self.dct_matrix_h.t(), torch.matmul(x, self.dct_matrix_w))

    # 定义一个方法，用于对输入数据进行高通滤波
    """
        在 HighDctFrequencyExtractor 类中，alpha 参数用于高通滤波操作。高通滤波的作用是保留图像的高频成分，高频成分主要体现图像的细节、纹理、边缘等微观信息。
        默认值 alpha = 0.05：这个值很小，表明在高通滤波时，少量的低频成分会被去除，而大部分高频成分会被保留。
        因为高频成分在频率域中通常集中在边缘区域，占比较小，设置较小的 alpha 能够精准地过滤掉低频部分，突出图像的细节特征。
    """
    def high_pass_filter(self, x, alpha):
        # 获取输入张量 x 的倒数第二个维度（高度）和最后一个维度（宽度）的大小
        h, w = x.shape[-2:]
        # 创建一个全为 1 的二维张量，形状与输入数据的高度和宽度相同，并放到与输入数据相同的设备上
        mask = torch.ones(h, w, device=x.device)
        # 根据 alpha 值计算需要保留的高度和宽度的像素数量
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        # 将掩码矩阵的左上角部分（低频部分）设置为 0，保留高频部分
        mask[:alpha_h, :alpha_w] = 0
        # 将输入数据与掩码矩阵逐元素相乘，实现高通滤波
        return x * mask

    # 定义前向传播方法，用于定义模型的计算流程
    def forward(self, x):
        # 对输入数据 x 进行二维 DCT 变换
        xq = self.dct_2d(x)
        # 对 DCT 变换后的结果进行高通滤波，保留高频部分
        xq_high = self.high_pass_filter(xq, self.alpha)
        # 对高通滤波后的结果进行二维逆 DCT 变换，将其转换回空间域
        xh = self.idct_2d(xq_high)

        # 获取输出张量 xh 的批次大小
        B = xh.shape[0]
        # 将每个批次的输出张量展平成一维向量，然后找出每个批次中的最小值
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        # 将每个批次的输出张量展平成一维向量，然后找出每个批次中的最大值
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        # 对输出张量进行归一化处理，将其值缩放到 [0, 1] 区间
        xh = (xh - min_vals) / (max_vals - min_vals)
        # 返回归一化后的输出张量
        return xh

if __name__ == '__main__':
    # 随机生成一个形状为 (1, 64, 32, 32) 的输入张量
    input_tensor = torch.rand(1, 64, 32, 32)
    # 创建 HighDctFrequencyExtractor 类的一个实例
    extractor = HighDctFrequencyExtractor()
    # 调用实例的 forward 方法，对输入张量进行处理
    output = extractor(input_tensor)
    print(input_tensor.size())  # 打印输入张量的形状
    print(output.size())  # 打印输出张量的形状
    print("公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")