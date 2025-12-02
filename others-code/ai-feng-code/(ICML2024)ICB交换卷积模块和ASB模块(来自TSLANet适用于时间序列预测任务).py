import lightning as L
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
adaptive_filter=True
import argparse
# 大家使用前先在自己的环境中安装下面这个包
# pip install lightning -i https://pypi.tuna.tsinghua.edu.cn/simple

'''
               ICML 2024  机器学习顶级会议 
时间序列数据以其固有的长程和短程依赖关系为特征，对分析应用程序构成了独特的挑战。
虽然基于 Transformer 的模型擅长捕获长距离依赖关系，但它们在噪声敏感性、
计算效率和与较小数据集的过度拟合方面面临限制。

作为回应，我们引入了一种新的时间序列轻量级自适应网络 （TSLANet），
作为不同时间序列任务的通用卷积模型。具体来说，我们提出了一种自适应频谱模块(ASB)，
利用傅里叶分析来增强特征表示并捕获长期和短期交互，同时通过自适应阈值减轻噪声。

此外，我们引入了一个交互式卷积块（ICB），并利用自我监督学习来改进 TSLANet 解码复杂时间模式的能力，
并提高其在不同数据集上的鲁棒性。我们的综合实验表明，
TSLANet 在时序分类、预测和异常检测等各种任务中优于最先进的模型，
展示了其在各种噪声水平和数据大小的弹性和适应性。

适用于：时序分类、预测、异常检测等所有时序任务通用模块
'''
class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1)) # * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape
        return x

if __name__ == '__main__':
    # 实例化ICB模型
    in_features = 16  # 输入特征数
    hidden_features = 32  # 隐藏层特征数
    drop = 0.1  # Dropout 概率
    model = ICB(in_features, hidden_features, drop)
    # 创建一个模拟输入数据张量 (batch_size, sequence_length, in_features)
    input= torch.randn(8, 10, in_features)
    # 执行前向传播
    output = model(input)
    # 打印输出张量的形状
    print("ICB输入张量形状:", input.shape)
    print("ICB输出张量形状:", output.shape)

    print("-------------------------")

    # 实例化Adaptive_Spectral_Block模型
    dim = 16  # 设置输入维度
    model = Adaptive_Spectral_Block(dim)
    # 创建一个模拟输入数据张量 (batch_size, sequence_length, dim)
    input = torch.randn(8, 32, dim)  # (batch_size, sequence_length, input_dim)
    # 执行前向传播
    output = model(input)
    # 打印输出张量的形状
    print("Adaptive_Spectral_Block输入张量形状:", input.shape)
    print("Adaptive_Spectral_Block输出张量形状:", output.shape)
