import torch
from torch import nn

from mamba_ssm import Mamba
# ；论文：https://arxiv.org/pdf/2403.20035
'''
PVMamba模块：

最近，以 Mamba 为代表的状态空间模型 （SSM） 已成为传统 CNN 和 Transformer 的强大竞争对手。

本文深入探讨了曼巴参数影响的关键要素，并在此基础上提出了一种超轻视觉曼巴UNet（UltraLight VM-UNet）。
具体来说，我们提出了一种并行处理Vision Mamba特征的方法，称为PVM Layer，
该方法在保持处理通道总数不变的情况下，以最低的计算复杂度实现了出色的性能。

我们在三个皮肤病变公共数据集上与几种最先进的轻量级模型进行了比较和消融实验，
结果表明，UltraLight VM-UNet在参数仅为0.049M，GFLOPs为0.060时，表现出同样强大的性能竞争力。

'''
class PVMamba(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2) #将B C L -->B L C
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


# ssh -p 27920 root@connect.westb.seetacloud.com
# NR34bWzc9YD2

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    models = PVMamba(input_dim=32,output_dim=32).cuda()
    input = torch.rand(3, 32, 64, 64).cuda()
    output = models(input)
    print('PVMLayer_input_size:',input.size())
    print('PVMLayer_output_size:',output.size())

