# --------------------------------------------------------
# 论文：DEA-Net: Single image dehazing based on detail enhanced convolution and content-guided attention
# https://arxiv.org/abs/2301.04805
# GitHub地址：https://github.com/cecret3350/DEA-Net/tree/main
# --------------------------------------------------------
'''
DEA-Net：基于细节增强卷积和内容引导注意力的单图像去雾 (IEEE TIP 2024顶会论文)
                            分析一下这篇能中顶会的摘要
单图像去雾是一个具有挑战性的病态问题，它从观察到的朦胧图像中估计出潜在的无雾图像。----描述本文解决的问题

一些现有的基于深度学习的方法致力于通过增加卷积的深度或宽度来提高模型性能。 ----夸一下现有深度学习在该问题上的已有效果（卷积模块作用）
低级表达：一些现有的基于深度学习的方法致力于单图像去雾取得了显著的效果。   ----夸之前的模型效果好，突出好在哪儿，卷积模块有效果

卷积神经网络（CNN）结构的学习能力仍未得到充分探索。  ----指明现有的卷积模块不足，暗示本文会提出一种比之前好用卷积模块

该文提出一种由细节增强卷积（DEConv）和内容引导注意力（CGA）组成的细节增强注意力块（DEAB）来促进特征学习，以提高去雾性能。
                                            ---本文的创新点，提出一个（卷积模块）和一个注意力机制，交代解决的任务
具体来说，DEConv将先验信息集成到正态卷积层中，以增强表示和泛化能力。
然后，通过使用重新参数化技术，DEConv被等效地转换为普通卷积，没有额外的参数和计算成本。
通过为每个通道分配唯一的空间重要性地图 （SIM），CGA 可以接收要素中编码的更多有用信息。  ---简单描述一下模块的基本实现及性能上（轻量和高效）的好处
'''
import math
import torch
from torch import nn
from einops.layers.torch import Rearrange

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_ad, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_rd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape
            if conv_weight.is_cuda:
                conv_weight_rd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
            else:
                conv_weight_rd = torch.zeros(conv_shape[0], conv_shape[1], 5 * 5)
            conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
            conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]
            conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:] * self.theta
            conv_weight_rd[:, :, 12] = conv_weight[:, :, 0] * (1 - self.theta)
            conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)
            out_diff = nn.functional.conv2d(input=x, weight=conv_weight_rd, bias=self.conv.bias,
                                            stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)

            return out_diff


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_hd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class DEConv(nn.Module):
    def __init__(self, dim):
        super(DEConv, self).__init__()

        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5
        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)
        return res

#DEConv_2二次创新模块：这里我只是对DEConv模块进行了初步的改进想法分享给大家，
# 1.使用了通道拼接再卷积压缩方法可以使得特征表达更充分
# 2.使用了门控权重缝合模块获取一个权重a，a*res1+(1-a)*res2突显res1和res2二者特征图之间的特征增强。
# 3.采用了残差连接，防止在细节增强特征提取过程中的信息丢失。out = x+a*res1+(1-a)*res2
#4.（大家下去可以实现一下）大家还可以对这四种卷积进行缝合：Conv2d_cd，Conv2d_hd，Conv2d_vd，Conv2d_ad
#  举例：比如增加一种通道注意力和空间注意力模块（CAM,SAM）注意力大家可以更换其它的，
#       增强这四种卷后的特征图在通道和空间上的信息特征。

class DEConv_2(nn.Module):
    # 模块名字大家根据自己解决的任务和网络模块的设计去命名（模块命名要突然它的作用）；
    # 别像我举例的这样去命名哈DEConv_2哈
    def __init__(self, dim):
        super(DEConv_2, self).__init__()

        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.conv1 =nn.Conv2d(dim*5,dim,1)
        self.sigmod = nn.Sigmoid()
    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        w1 = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5

        w2 = torch.cat([w1,w2,w3,w4,w5],dim=1)
        w2 = self.conv1(w2)
        res1 = nn.functional.conv2d(input=x, weight=w1, bias=b, stride=1, padding=1, groups=1)
        res2 = nn.functional.conv2d(input=x, weight=w2, bias=b, stride=1, padding=1, groups=1)
        a  =self.sigmod(res1+res2)
        out = x+a*res1+(1-a)*res2
        return out
# 双分支特征融合
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    models = DEConv_2(32).cuda()
    input = torch.rand(3, 32, 64, 64).cuda()
    output = models(input)
    print('input_size:',input.size())
    print('output_size:',output.size())
