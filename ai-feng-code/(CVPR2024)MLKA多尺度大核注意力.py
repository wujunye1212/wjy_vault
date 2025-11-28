import torch
import torch.nn as nn
from torch.nn import functional as F
# 代码：https://github.com/icandle/MAN
# 论文：https://arxiv.org/abs/2209.14145
'''
论文题目：面向单幅超分辨率图像的多尺度注意力网络   CVPR 2024顶会
摘要撰写分析：以后大家直接套模板5句话，任务背景+明确问题动机+点明创新点+简单介绍作用+做实验验证

ConvNets可以通过利用更大的感受野来与变压器（ViT）在高级任务中竞争。 ---夸卷积CNN具有竞争优势

为了在超分辨率下挖掘ConvNet的潜力，我们通过将经典的多尺度机制与新兴的大核注意力耦合， 
提出了一种多尺度注意力网络（MAN）。                       ----CNN不足：在超分图像任务处理（本文背景）上未发挥出它的优势
                                                    ---- 我们设计CNN网络,可以发挥它优势
                                                                                                         
具体而言，我们提出了多尺度大核注意力（MLKA）和门控空间注意力单元（GSAU）。  ---点明创新点，MLKA，GSAU

通过我们的MLKA，我们利用多尺度和门方案对大核注意力进行修改，以获得不同粒度级别的丰富注意力图，
从而聚合全局和局部信息，避免潜在的阻塞伪影。                
在GSAU中，我们整合了门机制和空间注意力，以去除不必要的线性层并聚合信息空间上下文。   ---- 简答介绍一下创新点的设计及作用

为了确认我们设计的有效性，我们通过简单地堆叠不同数量的MLKA和GSAU来评估具有多种复杂性的MAN算法。  ---通过对比实验，消融实验，我们的模块性能好
实验结果表明，我们的 MAN 的性能可以与 SwinIR 相当，并在最先进的性能和计算之间实现不同的权衡。

'''
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class SGAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        # Ghost Expand
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut
class MLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))

        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3)
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))
    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()

        x = self.norm(x)

        x = self.proj_first(x)

        a, x = torch.chunk(x, 2, dim=1)

        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)

        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)],
                      dim=1)
        x = self.proj_last(x * a) * self.scale + shortcut
        return x

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    ch = 10
    # 实例化模型对象
    model = MLKA(ch*3)
    # 生成随机输入张量
    input = torch.randn(1, ch*3, 32, 32)
    # 执行前向传播
    output = model(input)
    print('input_size:',input.size())
    print('output_size:',output.size())
