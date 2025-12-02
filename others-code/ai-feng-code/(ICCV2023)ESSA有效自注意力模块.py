import math
import torch
import torch.nn as nn
#论文： https://arxiv.org/pdf/2307.14010
'''
来自ICCV 2023 顶会论文
即插即用注意力模块： ESSA 有效自注意力模块

ESSA模块（Efficient SCC-kernel-based Self-Attention）是ESSAformer模型中的关键组件，
旨在提高高光谱图像超分辨率（HSI-SR）任务的计算效率和表现质量。
ESSA模块通过引入谱相关系数（SCC）计算特征之间的相似性，扩大感受野，捕获长距离的空间和光谱信息，
避免传统卷积网络因感受野限制而产生伪影。同时降低了计算复杂度并增强了模型在长距离依赖和小数据集上的表现，
特别适用于高光谱图像的超分辨率任务。它在提高图像恢复质量的同时，显著减少了计算负担，表现出色的视觉效果和定量结果。
ESSA模块的主要作用：
1.提高计算效率：通过核化自注意力机制，将传统自注意力计算的二次复杂度降低为线性复杂度,
             显著减少计算开销，尤其适用于高分辨率的高光谱图像。
2.增强长距离依赖建模：通过引入谱相关系数（SCC）计算特征之间的相似性，
           扩大感受野，捕获长距离的空间和光谱信息，避免传统卷积网络因感受野限制而产生的伪影。
3.提高数据效率与训练稳定性：SCC在计算注意力时，考虑了高光谱图像的光谱特性，
           能有效抗击阴影或遮挡等因素带来的光谱变化，提高模型在小数据集上的训练效率。
4.改善图像恢复质量：利用SCC的通道平移不变性和缩放不变性，减少由阴影、遮挡等造成的噪声，
           提高图像超分辨率恢复的质量，使得生成的高分辨率图像更加自然、平滑。



'''
class ESSAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        b,c,h,w=x.shape
        x = x.reshape(b,c,h*w).permute(0,2,1)
        b, N, C = x.shape
        qkv = self.lnqkv(x)
        qkv = torch.split(qkv, C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)
        t1 = v
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)

        attn = t1 + t2
        attn = self.ln(attn)
        x = attn.reshape(b,h,w,c).permute(0,3,1,2)
        return x

# 输入 N C H W,  输出 N C H W
if __name__ == "__main__":
    input =  torch.randn(1, 32, 64, 64)
    model = ESSAttn(32)
    output = model(input)
    print('input_size:',input.size())
    print('output_size:',output.size())
