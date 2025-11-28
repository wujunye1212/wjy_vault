import torch
import torch.nn as nn
from numpy import *
import random

'''
    论文地址：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05751.pdf
    论文题目：Efficient Frequency-Domain Image Deraining with Contrastive Regularization （ECCV 2024）
    中文题目： 利用对比正则化实现高效的频域图像去雨（ECCV 2024）
    讲解视频：https://www.bilibili.com/video/BV1V6UtYcEGH/
        频率对比正则化（Frequency Contrastive Regularization）：
           依据：FCR 使用GT作为正样本，雨纹模式图像作为负样本，同时使用 FADformer 的输出作为锚点。
           优点：1、正样本和负样本之间以及不同负样本之间的差异在频域中非常明显，这有利于对比学习。   	    	 
                2、通过快速傅里叶变换 (FFT)加速，几乎不影响训练速度。
'''

def sample_with_j(k, n, j):
    # 如果n大于等于k，抛出异常
    if n >= k:
        raise ValueError("n must be less than k.")
    # 如果j不在范围内，抛出异常
    if j < 0 or j > k:
        raise ValueError("j must be in the range 0 to k.")

    # 创建包含0到k的数字的列表
    numbers = list(range(k))

    # 确保j在数字列表中
    if j not in numbers:
        raise ValueError("j must be in the range 0 to k.")

    # 从数字列表中选择j
    sample = [j]

    # 从剩余的数字中选择n-1个
    remaining = [num for num in numbers if num != j]
    sample.extend(random.sample(remaining, n - 1))

    return sample

# -------------------FCR----------------------- #
# 频率对比正则化
# Frequency Contrastive Regularization
class FCR(nn.Module):
    def __init__(self, ablation=False):
        super(FCR, self).__init__()
        # L1损失函数
        self.l1 = nn.L1Loss()
        # 多个负样本的数量
        self.multi_n_num = 2

    def forward(self, a, p, n):
        # 计算a的频域表示
        a_fft = torch.fft.fft2(a)
        # 计算p的频域表示
        p_fft = torch.fft.fft2(p)
        # 计算n的频域表示
        n_fft = torch.fft.fft2(n)

        contrastive = 0
        # 遍历每个样本
        for i in range(a_fft.shape[0]):
            # 计算锚点与正样本的L1损失
            d_ap = self.l1(a_fft[i], p_fft[i])
            # 选择负样本
            for j in sample_with_j(a_fft.shape[0], self.multi_n_num, i):
                # 计算锚点与负样本的L1损失
                d_an = self.l1(a_fft[i], n_fft[j])
                # 累加对比损失
                contrastive += (d_ap / (d_an + 1e-7))
        # 归一化损失
        contrastive = contrastive / (self.multi_n_num * a_fft.shape[0])

        return contrastive

if __name__ == '__main__':
    # 假设输入张量的大小为 (batch_size, channels, height, width)
    # 创建随机输入张量
    a = torch.randn(4, 3, 32, 32)  # 锚点
    p = torch.randn(4, 3, 32, 32)  # 正样本
    n = torch.randn(4, 3, 32, 32)  # 负样本

    # 初始化FCR模块
    fcr = FCR()

    # 计算对比损失
    contrastive_loss = fcr(a, p, n)

    # 打印损失值
    print("Contrastive Loss:", contrastive_loss.item())


    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息
