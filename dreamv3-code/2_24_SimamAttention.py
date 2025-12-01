import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class SimamAttention(torch.nn.Module):

    def __init__(self, epsilon=1e-4):
        super(SimamAttention, self).__init__()
        self.activation = nn.Sigmoid()
        self.epsilon = epsilon

    def forward(self, x):

        # (B,C,H,W)
        batch_size, channels, height, width = x.size()

        # n = HW-1, 用于估计同通道的方差时的分母（“邻居数量”）
        n = width * height - 1

        # 计算每个通道的去均值平方: (B,C,H,W) - (B,C,1,1) == (B,C,H,W), (t−μ)^2, 其中 t 表示该通道任一空间位置的值。它衡量“该像素与通道平均值”的差异（显著性）
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # step1: 对同一通道在空间上求和, ∑(x−μ)^2: (B,C,H,W)-sum->(B,C,1,1)
        # step2: 把x_minus_mu_square求和除以 n=HW−1, 得到近似的同通道方差估计 σ^2; 此外,加 ε 提高数值稳定性, 避免 σ2 ≈ 0 时分母过小
        # step3: 以上step1 和 step2 计算的是分子: 4(σ2 + λ); * 4; 这是 SimAM 推导里对能量函数进行归一化后得到的系数，可理解为把数值范围缩放到合适区间。
        # step4:  x_minus_mu_square 是事先计算好的分子
        attention = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.epsilon)) + 0.5

        # 通过sigmoid函数
        return x * self.activation(attention)



if __name__ == '__main__':
    # (B,C,H,W)
    x1 = torch.randn(1, 64, 224, 224)
    B, C, H, W = x1.size()

    # 定义 AttentionTSSA
    Model = SimamAttention()

    # 执行 AttentionTSSA
    out = Model(x1)
    print(out.shape)

