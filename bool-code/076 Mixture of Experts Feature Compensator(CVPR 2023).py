import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Learning_a_Sparse_Transformer_Network_for_Effective_Image_Deraining_CVPR_2023_paper.pdf
    论文题目：Learning A Sparse Transformer Network for Effective Image Deraining（CVPR 2023）
    中文题目：学习稀疏的变压器网络以实现有效的图像去雨 （CVPR 2023）
    讲解视频：https://www.bilibili.com/video/BV1iuB7Y5EU2/
        混合专家特征补偿器（Mixture of Experts Feature Compensator，MEFC）：
            设计目的：为了整合稀疏性，对联合数据的信息稀疏性进行统一协同探索，实现更好的去雨。
            理论支撑：选择多个稀疏 CNN 操作形成并行层，包括 3×3 感受野的平均池化、1×1 核的可分离卷积层、3×3、5×5、7×7 的常规卷积层以及 3×3、5×5、7×7 的空洞卷积层。
"""

Operations = [
    'sep_conv_1x1',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'dil_conv_7x7',
    'avg_pool_3x3'
]

OPS = {
    'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'sep_conv_1x1' : lambda C, stride, affine: SepConv(C, C, 1, stride, 0, affine=affine),
    'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'dil_conv_7x7' : lambda C, stride, affine: DilConv(C, C, 7, stride, 6, 2, affine=affine),
}

class ReLUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),)

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),)

    def forward(self, x):
        return self.op(x)

class OperationLayer(nn.Module):
    def __init__(self, C, stride):
        super(OperationLayer, self).__init__()
        self._ops = nn.ModuleList()
        for o in Operations:
            op = OPS[o](C, stride, False)
            self._ops.append(op)

        self._out = nn.Sequential(nn.Conv2d(C * len(Operations), C, 1, padding=0, bias=False), nn.ReLU())

    def forward(self, x, weights):
        weights = weights.transpose(1, 0)
        states = []
        for w, op in zip(weights, self._ops):
            states.append(op(x) * w.view([-1, 1, 1, 1]))
        return self._out(torch.cat(states[:], dim=1))

class GroupOLs(nn.Module):
    def __init__(self, steps, C):
        super(GroupOLs, self).__init__()
        self.preprocess = ReLUConv(C, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._ops = nn.ModuleList()
        self.relu = nn.ReLU()
        stride = 1

        for _ in range(self._steps):
            op = OperationLayer(C, stride)
            self._ops.append(op)

    def forward(self, s0, weights):
        s0 = self.preprocess(s0)
        for i in range(self._steps):
            res = s0
            s0 = self._ops[i](s0, weights[:, i, :])
            s0 = self.relu(s0 + res)
        return s0

class OALayer(nn.Module):
    def __init__(self, channel, k, num_ops):
        super(OALayer, self).__init__()
        self.k = k
        self.num_ops = num_ops
        self.output = k * num_ops
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Linear(channel, self.output * 2),
            nn.ReLU(),
            nn.Linear(self.output * 2, self.k * self.num_ops))

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc(y)
        y = y.view(-1, self.k, self.num_ops)
        return y

## Mixture of Experts Feature Compensator (MEFC)
class subnet(nn.Module):
    def __init__(self, dim, layer_num=1, steps=4):
        super(subnet, self).__init__()

        self._C = dim
        self.num_ops = len(Operations)
        self._layer_num = layer_num
        self._steps = steps

        self.layers = nn.ModuleList()
        for _ in range(self._layer_num):
            attention = OALayer(self._C, self._steps, self.num_ops)
            self.layers += [attention]
            layer = GroupOLs(steps, self._C)
            self.layers += [layer]

    def forward(self, x):

        for _, layer in enumerate(self.layers):
            if isinstance(layer, OALayer):
                weights = layer(x)
                weights = F.softmax(weights, dim=-1)
            else:
                x = layer(x, weights)

        return x

if __name__ == '__main__':
    # 创建一个随机输入张量作为示例
    input_tensor = torch.randn(1, 64, 32, 32)  # 假设输入尺寸为 [batch_size, channels, height, width]

    mdcr = subnet(dim=64)

    output_tensor = mdcr(input_tensor)

    # 打印输入和输出的形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")