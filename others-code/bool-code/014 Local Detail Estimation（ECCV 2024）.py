import torch
import torch.nn as nn

"""
论文地址：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf
论文题目：SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution（ECCV 2024）
讲解视频：https://www.bilibili.com/video/BV1Wrwwe6EjR/

    局部细节估计 (Local Detail Estimation, LDE)
       为了获得高频成分，首先使用一个3×3深度可分离卷积来从输入特征Y中编码局部信息Yh。
       然后，使用两个带有隐藏GELU激活函数的1×1卷积生成增强的局部特征Yd。
       通过这种方式，LDE分支能够有效地捕捉到图像中的局部细节，这些细节对于提高超分辨率重建的质量至关重要。
"""
class LDE(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()  # 调用父类nn.Module的初始化方法
        hidden_dim = int(dim * growth_rate)  # 计算隐藏层维度，基于输入维度和增长因子
        self.conv_0 = nn.Sequential(  # 定义第一个卷积序列
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),  # 深度可分离卷积，使用分组卷积来减少参数
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)      # 点卷积，用于通道融合
        )
        self.act = nn.GELU()  # 使用GELU激活函数
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)  # 最后一个点卷积，将隐藏层维度还原到输入维度

    def forward(self, x):
        x = self.conv_0(x)  # 应用第一个卷积序列
        x = self.act(x)  # 应用激活函数
        x = self.conv_1(x)  # 应用最后一个点卷积
        return x  # 返回最终输出

# 输入 N C H W, 输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1, 32, 256, 256)  # 创建一个随机输入张量，形状为 (1, 32, 256, 256)
    model = LDE(dim=32)  # 实例化LDE模型，指定输入维度为32
    output = model(input)  # 将输入传递给模型，得到输出
    print('input_size:', input.size())  # 打印输入张量的大小
    print('output_size:', output.size())  # 打印输出张量的大小