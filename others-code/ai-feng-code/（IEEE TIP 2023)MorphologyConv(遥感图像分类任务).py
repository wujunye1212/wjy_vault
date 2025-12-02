import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#论文：https://ieeexplore.ieee.org/document/10050427

'''
论文题目：跨场景高光谱图像分类的单源域扩展网络
本论文动机是目前图像分类任务存在以下之处：
1.领域自适应实用性差
虽然深度领域自适应的方法在跨场景高光谱分类任务中取得长足发展，
但在实际应用中仍存在一些问题，当平台计算资源低、实时性要求较高时，
要求目标场景数据实时获取、实时预测。领域自适应方法都无法满足这一要求。
2.缺乏考虑光谱信息
领域泛化在CV中有一定的发展，但传统的领域泛化关注空间信息，
忽略光谱信息，不适用于高光谱图像。例如学习随机化风格的方法只关注空间层面的多样性，
光谱多样性差，而高光谱图像是空-谱合一的多维数据，需要做到空间和谱间的多维度多样性。

本文总结：
1.设计形态编码器提取具有域不变性特征，确保了生成样本的可靠性
2.设计空间-高光谱信息联合的语义编码器，确保了生成样本的有效性；

'''
class MorphologyConv(nn.Module):#形态学卷积模块
    '''
    Base class for morpholigical operators
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure.
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(),
        ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(MorphologyConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type

        self.weight = nn.Parameter(torch.ones(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)

        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))

        # erosion
        weight = self.weight.view(self.out_channels, -1)  # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)

        if self.type == 'erosion2d':
            x = weight - x  # (B, Cout, Cin*kH*kW, L)
        elif self.type == 'dilation2d':
            x = weight + x  # (B, Cout, Cin*kH*kW, L)
        else:
            raise ValueError

        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False)  # (B, Cout, L)
        else:
            x = torch.logsumexp(x * self.beta, dim=2, keepdim=False) / self.beta  # (B, Cout, L)

        if self.type == 'erosion2d':
            x = -1 * x

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)

        return x

class Dilation2d(MorphologyConv):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'dilation2d')

class Erosion2d(MorphologyConv):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'erosion2d')

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    #MorphologyConv形态学卷积模块：
    models = MorphologyConv(in_channels=3,out_channels=3,type='dilation2d').cuda()
    input = torch.randn(1, 3, 32, 32).cuda()
    output = models(input)
    print('input_size:',input.size())
    print('output_size:',output.size())