import torch
import torch.nn as nn

'''
来自CVPR 2025顶会  9行代码砍掉归一化层！性能反而更强！

即插即用模块 : Dynamic Tanh（DyT） 它可以取代归一化层 Layer Norm 
             含一个创新DyTConv卷积模块
主要内容：
归一化层在现代神经网络中无处不在，并长期被认为是必不可少的。
本研究表明，无归一化的 Transformer 也能实现相同甚至更好的性能，而仅依赖一种极其简单的技术。
我们提出了一种名为 Dynamic Tanh (DyT) 的逐元素运算，其定义为：DyT(x)=tanh(αx)

DyT 可以直接替换 Transformer 中的归一化层， Layer Norm (LN)。
DyT 的灵感来自于我们对 Transformer 归一化层的观察，即 Layer Norm 在很多情况下会生成类似于 tanh 的 S 形输入-输出映射。
DyT 通过学习一个合适的缩放因子 α 来模拟这种映射，并利用 tanh 函数的非线性特性来抑制极端值，而无需计算激活值的统计信息。

我们在多个不同的任务场景下验证了 DyT 的有效性，
包括图像识别、文本生成、有监督学习、自监督学习、计算机视觉以及自然语言处理等领域。
实验结果表明，采用 DyT 的 Transformer 可以匹配甚至超越使用归一化层的 Transformer，并且大多数情况下不需要额外调整超参数。

这些研究结果挑战了传统观点，即归一化层在深度神经网络中是不可或缺的，并提供了新的见解，帮助我们更好地理解归一化层在深度学习中的作用。
Dynamic Tanh (DyT) 是本文提出的一种替代 Transformer 归一化层（ Layer Norm ）的方法。
其主要作用包括：
 1.去归一化：通过 DyT 取代归一化层，使得 Transformer 网络在没有归一化层的情况下仍然能够稳定训练并获得高性能。
 2.非线性激活：DyT 通过 tanh(αx) 形式，使得输入特征的映射过程类似于 Layer Norm 在训练过程中形成的S形曲线。
 3.计算高效：DyT 避免了 Layer Norm 需要计算均值和方差的开销，提高了训练和推理的效率。
'''

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=True, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class DyTConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, shape,padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1,shape=[], p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.DyT = DynamicTanh(shape)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, DynamicTanh  and activation to input tensor."""
        x = self.conv(x)
        out = self.bn(x*self.DyT(x)) #大家可以合理玩一下这个DyT模块，但是不要直接替换bn批标准化，不然容易造成训练不稳定。
        return self.act(out)

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    input = torch.randn(1,32,128, 128)  # 创建一个形状为 (1,32,128, 128)
    DyT = DynamicTanh([32,128,128])
    output = DyT(input)  # 通过 DyTConv 模块计算输出
    print('DyT_Input size:', input.size())  # 打印输入张量的形状
    print('DyT_Output size:', output.size())  # 打印输出张量的形状


    input_tensor = torch.randn(1,32,128, 128)  # 创建一个形状为 (1,32,128, 128)
    # 创建 DyTConv 模块实例，输入通道数为32，输出通道数为 64，卷积核为1，步长为1。
    # module =DyTConv(32,64,1,1,[64,128,128])
    module =DyTConv(32,64,3,2,[64,64,64])
    output_tensor = module(input_tensor)  # 通过 DyTConv 模块计算输出
    print('DyTConv_Input size:', input_tensor.size())  # 打印输入张量的形状
    print('DyTConv_Output size:', output_tensor.size())  # 打印输出张量的形状