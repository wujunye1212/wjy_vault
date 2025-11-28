# 论文：https://arxiv.org/abs/2212.01209#
# https://github.com/Zero-coder/FECAM/blob/main/layers/dctnet.py

"""
                   FECAM：用于时间序列预测的频率增强信道注意力机制
时间序列预测长期以来一直是一项挑战，因为现实世界的信息涉及多种场景（例如能源、天气、交通、经济、地震预警）。
然而，一些主流预测模型的结果与真实值有较大偏差。我们认为这是因为这些模型缺乏捕捉频率信息的能力，
而这种信息在真实世界的数据集中丰富存在。

目前，主流的频率信息提取方法基于傅里叶变换（FT）。然而，使用FT存在问题，即所谓的吉布斯现象（Gibbs phenomenon）。
如果序列两端的值差异显著，则在两端附近会出现振荡近似，并引入高频噪声。

因此，我们提出了一种新的基于离散余弦变换（DCT）的频率增强通道注意力机制，
该机制能够内在地避免傅里叶变换过程中因周期性问题导致的高频噪声，即避免吉布斯现象。
我们展示了这种网络在六个真实世界数据集上具有极高的泛化效果，并达到了最先进的性能水平。
此外，我们还证明了这种频率增强的通道注意力机制可以灵活地应用于不同的网络中。
"""

import torch.nn as nn
import numpy as np
import torch
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r


    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
        return t.real


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # Vc = torch.fft.rfft(v, 1, onesided=False)
    Vc = rfft(v, 1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V
class DCT_Attention(nn.Module):
    def __init__(self, channel):
        super(DCT_Attention, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1) #innovation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.Sigmoid()
        )
        # self.dct_norm = nn.LayerNorm([512], eps=1e-6)

        self.dct_norm = nn.LayerNorm([channel], eps=1e-6)  # for lstm on length-wise
        # self.dct_norm = nn.LayerNorm([36], eps=1e-6)#for lstm on length-wise on ill with input =36

    def forward(self, x):
        b, c, l = x.size()  # (B,C,L) (32,96,512)
        list = []
        for i in range(c):
            freq = dct(x[:, i, :])
            # print("freq-shape:",freq.shape)
            list.append(freq)

        stack_dct = torch.stack(list, dim=1)
        stack_dct = torch.tensor(stack_dct)
        '''
        for traffic mission:f_weight = self.dct_norm(f_weight.permute(0,2,1))#matters for traffic datasets
        '''

        lr_weight = self.dct_norm(stack_dct)
        lr_weight = self.fc(stack_dct)
        lr_weight = self.dct_norm(lr_weight)

        # print("lr_weight",lr_weight.shape)
        return x * lr_weight  # result
if __name__ == '__main__':
    tensor = torch.rand(8, 7, 96)
    dct_model = DCT_Attention(96)
    result =dct_model(tensor)
    print("input.shape:", tensor.shape)
    print("output.shape:", result.shape)