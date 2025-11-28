import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from typing import List
from torch import Tensor
import os
import copy
from mmcv.cnn import build_norm_layer
from math import log
import numpy


class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   build_norm_layer(norm_layer, channel)[1])
    def forward(self, x):
        out = self.block(x) # (B,C,H,W)--三层卷积-->(B,C,H,W)
        return out


class Scharr(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Scharr, self).__init__()
        # 定义Scharr滤波器
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze \
            (0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze \
            (0).unsqueeze(0)
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        # 将Sobel滤波器分配给卷积层
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1) # (1,1,3,3)-repeat->(C,1,3,3)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1) # (1,1,3,3)-repeat->(C,1,3,3)
        self.norm = build_norm_layer(norm_layer, channel)[1]
        self.act = act_layer()
        self.conv_extra = Conv_Extra(channel, norm_layer, act_layer)

    def forward(self, x):
        # show_feature(x)
        # 应用卷积操作
        edges_x = self.conv_x(x) # (B,C,H,W)-->(B,C,H,W)
        edges_y = self.conv_y(x) # (B,C,H,W)-->(B,C,H,W)
        # 计算边缘和高斯分布强度（可以选择不同的方式进行融合，这里使用平方和开根号）
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2) # (B,C,H,W)-->(B,C,H,W)
        scharr_edge = self.act(self.norm(scharr_edge)) # BatchNorm and relu
        out = self.conv_extra(x + scharr_edge) # 添加残差连接, 并通过三层卷积进行融合：(B,C,H,W)-->(B,C,H,W)

        return out


class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma) # 高斯核：(1,1,5,5)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone() # 参数固定
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1) # (C,1,5,5)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x) # 执行具有高斯核的卷积操作: (B,C,H,W)-->(B,C,H,W)
        gaussian = self.act(self.norm(edges_o)) # BatchNorm and relu
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian) # 添加残差连接, 并通过三层卷积进行融合：(B,C,H,W)-->(B,C,H,W)
        else:
            out = gaussian
        return out

    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0) # (5,5)--unsqueeze-->(1,5,5)--unsqueeze-->(1,1,5,5)
        return kernel / kernel.sum()


class LFEA(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(LFEA, self).__init__()
        self.channel = channel
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = self.block = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, dilation=1, bias=False),
            build_norm_layer(norm_layer, channel)[1],
            act_layer())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = build_norm_layer(norm_layer, channel)[1]

    def forward(self, c, att):
        att = c * att + c  # 矩阵点乘, 然后添加残差连接: (B,C,H,W)
        att = self.conv2d(att) # 卷积层: (B,C,H,W)-->(B,C,H,W), 按理说到这, 就已经得到 EGA模块的输出了, 以下是LEG(包含EGA)的额外计算
        wei = self.avg_pool(att) # 在空间层面执行平均池化: (B,C,H,W)-->(B,C,1,1)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) # 在通道方向进行局部特征提取: (B,C,1,1)-squ->(B,C,1)-trans->(B,1,C)-conv1d->(B,1,C)-trans->(B,C,1)-unsqu->(B,C,1,1)
        wei = self.sigmoid(wei) # 生成权重: (B,C,1,1)
        x = self.norm(c + att * wei) # 使用权重调整EGA的输出, 并与最初的输入完成残差操作： (B,C,H,W)

        return x



class EGA_Module(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 mlp_ratio,
                 drop_path,
                 act_layer,
                 norm_layer
                 ):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)]

        self.mlp = nn.Sequential(*mlp_layer)
        self.LFEA = LFEA(dim, norm_layer, act_layer)

        if stage == 0:
            self.Scharr_edge = Scharr(dim, norm_layer, act_layer)
        else:
            self.gaussian = Gaussian(dim, 5, 1.0, norm_layer, act_layer)
        self.norm = build_norm_layer(norm_layer, dim)[1]

    def forward(self, x: Tensor) -> Tensor:
        # show_feature(x)
        if self.stage == 0:
            att = self.Scharr_edge(x) # 方向感知的scharr滤波器：(B,C,H,W)-->(B,C,H,W)
        else:
            att = self.gaussian(x) # 基于高斯先验的特征细化：(B,C,H,W)-->(B,C,H,W)
        x_att = self.LFEA(x, att) # x:(B,C,H,W); att:(B,C,H,W); x_att: (B,C,H,W)
        x = x + self.norm(self.drop_path(self.mlp(x_att))) # (B,C,H,W)
        return x




if __name__ == '__main__':

    # (B,C,H,W)
    x1 = torch.randn(1, 64, 224, 224)
    B, C, H, W = x1.size() # shape
    depth = 1  # 模型深度, 简单来说, 这是你模型中的第几层?

    # 定义 EGA
    Model = EGA_Module(
                dim=C,
                stage=depth,
                mlp_ratio=2,
                drop_path=0.,
                norm_layer=dict(type='BN', requires_grad=True),
                act_layer=nn.ReLU
            )

    # 执行 EGA
    out = Model(x1)
    print(out.shape)