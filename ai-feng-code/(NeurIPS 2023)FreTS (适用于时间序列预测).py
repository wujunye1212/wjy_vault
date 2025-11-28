import torch
import torch.nn as nn
import torch.nn.functional as F
# https://github.com/aikunyi/FreTS/tree/main
# https://arxiv.org/abs/2311.06184

'''
题目： 频域 MLP 是时间序列预测中更有效的学习器    
 
FreTS:基于频域 MLP 的简单而有效的架构    NeurIPS顶会 2023
 
时间序列预测在不同的行业中发挥了关键作用，包括金融、交通、能源和医疗保健领域。
虽然现有文献已经设计了许多基于 RNN、GNN 或 Transformer 的复杂架构，
但提出了另一种基于多层感知器 （MLP） 的方法，具有结构简单、复杂度低和卓越性能。

然而，大多数基于 MLP 的预测方法都存在逐点映射和信息瓶颈，这在很大程度上阻碍了预测性能。
为了克服这个问题，我们探索了在频域中应用 MLP 进行时间序列预测的新方向。
我们研究了频域 MLP 的学习模式，并发现了它们有利于预测的两个固有特性，
（i） 全局视图：频谱使 MLP 拥有完整的信号视图，更容易学习全局依赖关系，
（ii） 能量压缩：频域 MLP 集中在具有紧凑信号能量的频率分量的较小关键部分。

然后，我们提出了 FreTS，这是一种基于频域 MLP 的简单而有效的架构，用于时间序列预测。

FreTS 主要涉及两个阶段，
（i） 域转换，将时域信号转换为频域复数;
（ii） 频率学习，执行我们重新设计的 MLP，用于学习频率分量的实部和虚部。

通过广泛实验表明(包括 7 个短期预测基准和 6 个长期预测基准），我们始终优于最先进的方法。

'''
class Configs:
    def __init__(self):
        self.pred_len = 32  # 预测长度
        self.enc_in = 16  # 通道数
        self.seq_len = 32  # 输入序列长度
        self.channel_independence = '1'  # 是否独立处理通道
class FreTS(nn.Module):
    def __init__(self, configs):
        super(FreTS, self).__init__()
        self.embed_size = 128 #embed_size
        self.hidden_size = 256 #hidden_size
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in #channels
        self.seq_length = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x)
        bias = x
        # [B, N, T, D]
        if self.channel_independence == '1':
            x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x

if __name__ == '__main__':
    # 初始化配置
    configs = Configs()
    # 实例化FreTS模块
    frets = FreTS(configs)
    # 创建一个随机输入张量，形状为[Batch, Input length, Channel]
    input = torch.randn(8, configs.seq_len, configs.enc_in)
    # 执行前向传播
    output = frets(input)
    # 打印输出张量的形状
    print("Input shape:", input.shape)
    print("Output shape:", output.shape)