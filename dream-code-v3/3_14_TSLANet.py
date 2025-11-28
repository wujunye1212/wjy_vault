import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from timm.models.layers import trunc_normal_

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02) # (C,2):可训练的权重, 存储了频率分量的实部和虚部
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02) # 随机初始化实部和虚部（从正态分布中采样）,然后乘以一个小值 0.02 来避免初始权重过大。
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1)) # * 0.5)
        self.adaptive_filter = True

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape # (B,N/2+1,C)

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1) # 计算输入的频率分量的能量(强度),能量被定义为频率分量的平方的和,表示了每个频率分量的强度. 最后沿着通道方向求和: (B,N/2+1,C)-->(B,N/2+1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # (B,N/2+1). 将频率分量的能量展平,以便计算全局的中位数能量
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # (B,1). 计算展平后的能量的中位数。中位数被用作基准,以确定哪些频率分量能量大于中位数。
        median_energy = median_energy.view(B, 1)  # (B,1).  Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6) # (B,N/2+1). 能量通过中位数进行归一化,得到归一化能量
        # 根据归一化后的能量生成自适应掩码(B,N/2+1). 这个掩码将用于对频率分量进行筛选,过滤掉较低能量的部分。
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1) # (B,N/2+1,1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # 应用FFT, 将时间域数据转换为频域数据
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho') # 生成频域表示, 返回一个复数张量, 只保留正频率项: (B,N,C)-->(B,N/2+1,C)

        # 从原始频域信号中学习全局信息
        weight = torch.view_as_complex(self.complex_weight) # 将complex_weight转换为复数表示: (C,2)-->(C,)
        x_weighted = x_fft * weight # 使用weight对频域x_fft加权: (B,N/2+1,C)

        # 计算掩码,并与频域数据相乘, 去除噪声, 保留重要信息
        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft) # 生成一个自适应高频掩码,这个掩码将用于对频率分量进行筛选,过滤掉较低能量的部分,只保留高能量: (B,N/2+1,C)-->(B,N/2+1,1)
            x_masked = x_fft * freq_mask.to(x.device) # 将掩码应用到频域数据,得到低频分量: (B,N/2+1,C) * (B,N/2+1,1) == (B,N/2+1,C)

        # 从过滤后的频域信号中学习局部信息
        weight_high = torch.view_as_complex(self.complex_weight_high) # 将complex_weight_high转换为复数表示: (C,2)-->(C,)
        x_weighted2 = x_masked * weight_high # 使用weight_high对频域x_masked加权: (B,N/2+1,C)

        # 局部信息与全局信息融合
        x_weighted += x_weighted2 #将低频分量加权后的结果与原始加权结果相加，以增强特征表示: (B,N/2+1,C) + (B,N/2+1,C) == (B,N/2+1,C)

        # 通过Inverse FFT,将频域数据转换为时间域数据
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho') # 对加权后的频域数据进行逆傅里叶变换,将数据从频域转换回时间域，恢复原始的时间序列表示: (B,N/2+1,C)--irfft-->(B,N,C)

        x = x.to(dtype) # 将数据转换回原始输入的精度
        x = x.view(B, N, C)  # Reshape back to original shape

        return x



if __name__ == '__main__':
    #  (B,T,C)
    x1 = torch.randn(10,96,64).to(device)

    Model = Adaptive_Spectral_Block(dim=64).to(device)

    out = Model(x1) # (B,T,C)--> (B,T,C)
    print(out.shape)