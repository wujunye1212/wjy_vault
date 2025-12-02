import torch.nn as nn
import torch.nn.functional as F
import torch
from timm.layers import CondConv2d

'''
 ParameterNet: Parameters Are All You Need
 
大规模视觉预训练已经显著提高了大型视觉模型的表现。
然而，我们观察到低FLOPs（浮点运算次数）模型存在的问题，
即现有低FLOPs模型无法从大规模预训练中受益。

在这篇论文中，我们引入了一种新的设计理念，称为ParameterNet（参数网络），
旨在增加大规模视觉预训练模型的参数数量，同时最小化FLOPs的增加。

我们利用动态卷积(DynamicConv)将额外的参数加入到网络中，只带来边际的FLOPs增加。
ParameterNet方法使得低FLOPs网络能够从大规模视觉预训练中受益。
此外，我们将ParameterNet的概念扩展到语言领域，以提高推理结果的同时保持推理速度不变。

'''
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class DynamicConv(nn.Module):
    """ Dynamic Conv layer
    """
    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding='', dilation=1,
                 groups=1, bias=False, num_experts=4):
        super().__init__()
        # print('+++', num_experts)
        self.routing = nn.Linear(in_features, num_experts)
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation,
                                    groups, bias, num_experts)

    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        x = self.cond_conv(x, routing_weights)
        return x
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block =DynamicConv(32,32)
    input = torch.rand(1, 32, 64, 64)
    output = block(input)
    print("input.shape:", input.shape)
    print("output.shape:",output.shape)