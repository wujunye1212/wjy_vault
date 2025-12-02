import torch
import torch.nn as nn
# 论文：https://www.sciencedirect.com/science/article/abs/pii/S1566253523001860
# https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/PSFM.jpg

'''
多尺度图像特征融合即插即用模块：PSFM      2023 SCI一区
本文内容总结：
图像融合旨在将源图像的互补特性整合到单个融合图像中，以更好地服务于人类视觉观察和机器视觉感知。
然而，现有的大多数图像融合算法主要侧重于提高融合图像的视觉吸引力。
本文介绍了一种名为PSFM 的新型红外和可见光图像融合网络。

PSFM  利用渐进语义注入和场景保真度约束来优化融合图像，提升其对高级视觉任务的适用性。
稀疏语义感知分支和语义注入模块确保融合特征满足高级视觉任务需求，
而场景恢复分支中的场景保真度路径确保融合图像包含源图像的完整信息。

对比实验表明，PSFM  在视觉效果和高级语义方面优于现有图像级融合方法，
并能充分利用多模态数据和最新单模态数据，实现卓越的性能。

结论：
本文提出了一种实用的多尺度图像融合模块，称为PSFM  ，
并能充分利用多模态数据和最新单模态数据，实现卓越的性能。
在各种cv任务上是通用即插即用模块
'''
class BBasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BBasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

class GEFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(GEFM, self).__init__()
        self.RGB_K = BBasicConv2d(out_C, out_C, 3, padding=1)
        self.RGB_V = BBasicConv2d(out_C, out_C, 3, padding=1)
        self.Q = BBasicConv2d(in_C, out_C, 3, padding=1)
        self.INF_K = BBasicConv2d(out_C, out_C, 3, padding=1)
        self.INF_V = BBasicConv2d(out_C, out_C, 3, padding=1)
        self.Second_reduce = BBasicConv2d(in_C, out_C, 3, padding=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        Q = self.Q(torch.cat([x, y], dim=1))
        RGB_K = self.RGB_K(x)
        RGB_V = self.RGB_V(x)
        m_batchsize, C, height, width = RGB_V.size()
        RGB_V = RGB_V.view(m_batchsize, -1, width * height)
        RGB_K = RGB_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        RGB_Q = Q.view(m_batchsize, -1, width * height)
        RGB_mask = torch.bmm(RGB_K, RGB_Q)
        RGB_mask = self.softmax(RGB_mask)
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, height, width)
        RGB_refine = self.gamma1 * RGB_refine + y

        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)
        INF_V = INF_V.view(m_batchsize, -1, width * height)
        INF_K = INF_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        INF_Q = Q.view(m_batchsize, -1, width * height)
        INF_mask = torch.bmm(INF_K, INF_Q)
        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height, width)
        INF_refine = self.gamma2 * INF_refine + x

        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))
        return out

class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=2):
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BBasicConv2d(mid_C * i, mid_C, 3, padding=1))

        self.fuse = BBasicConv2d(in_C + mid_C, out_C, 3, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for i in self.denseblock:
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)

class PSFM(nn.Module):
    def __init__(self, Channel):
        super(PSFM, self).__init__()
        self.RGBobj = DenseLayer(Channel, Channel)
        self.Infobj = DenseLayer(Channel, Channel)
        self.obj_fuse = GEFM(Channel * 2, Channel)

    def forward(self, data):
        rgb, depth = data
        rgb_sum = self.RGBobj(rgb)
        Inf_sum = self.Infobj(depth)
        out = self.obj_fuse(rgb_sum, Inf_sum)
        return out
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = PSFM(Channel=32)
    input1 = torch.rand(1, 32, 64, 64)
    input2 = torch.rand(1, 32, 64, 64)
    output = block([input1, input2])
    print('input1_size:', input1.size())
    print('input2_size:', input2.size())
    print('output_size:', output.size())
