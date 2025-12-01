import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import functools
import math
import timm


class GlobalExtraction(nn.Module):
  def __init__(self,dim = None):
    super().__init__()
    self.avgpool = self.globalavgchannelpool
    self.maxpool = self.globalmaxchannelpool
    self.proj = nn.Sequential(
        nn.Conv2d(2, 1, 1,1),
        nn.BatchNorm2d(1)
    )
  def globalavgchannelpool(self, x):
    x = x.mean(1, keepdim = True)
    return x

  def globalmaxchannelpool(self, x):
    x = x.max(dim = 1, keepdim=True)[0]
    return x

  def forward(self, x):
    x_ = x.clone()
    x = self.avgpool(x) #(B,1,H,W)
    x2 = self.maxpool(x_) #(B,1,H,W)

    cat = torch.cat((x,x2), dim = 1) # (B,2,H,W)

    proj = self.proj(cat) # (B,2,H,W)-->(B,1,H,W)
    return proj

class ContextExtraction(nn.Module):
  def __init__(self, dim, reduction = None):
    super().__init__()
    self.reduction = 1 if reduction == None else 2

    self.dconv = self.DepthWiseConv2dx2(dim)
    self.proj = self.Proj(dim)

  def DepthWiseConv2dx2(self, dim):
    dconv = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 1,
              groups = dim),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True),
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 2,
              dilation = 2),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True)
    )
    return dconv

  def Proj(self, dim):
    proj = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim //self.reduction,
              kernel_size = 1
              ),
        nn.BatchNorm2d(num_features = dim//self.reduction)
    )
    return proj
  def forward(self,x):
    x = self.dconv(x) # (B,C,H,W)-->  (B,C,H,W)
    x = self.proj(x)  # (B,C,H,W)-->  (B,C,H,W)
    return x

class MultiscaleFusion(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.local= ContextExtraction(dim)
    self.global_ = GlobalExtraction()
    self.bn = nn.BatchNorm2d(num_features=dim)

  def forward(self, x, g,):
    x = self.local(x)  # 提取局部特征: (B,C,H,W)-->(B,C,H,W)
    g = self.global_(g) # 提取全部特征: (B,C,H,W)-->(B,1,H,W)
    fuse = self.bn(x + g) # 融合局部和全局特征: (B,C,H,W) + (B,1,H,W) == (B,C,H,W)
    return fuse


class MultiScaleGatedAttn(nn.Module):
    # Version 1
  def __init__(self, dim):
    super().__init__()
    self.multi = MultiscaleFusion(dim)
    self.selection = nn.Conv2d(dim, 2,1)
    self.proj = nn.Conv2d(dim, dim,1)
    self.bn = nn.BatchNorm2d(dim)
    self.bn_2 = nn.BatchNorm2d(dim)
    self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=dim, out_channels=dim,
                  kernel_size=1, stride=1))

  def forward(self,x,g):
    x_ = x.clone() # (B,C,H,W)
    g_ = g.clone() # (B,C,H,W)

    # 1、多尺度特征融合
    multi = self.multi(x, g) # x:(B,C,H,W), g:(B,C,H,W)--> multi: (B,C,H,W)

    # 2、空间选择
    multi = self.selection(multi) # 投影到2个通道,这相当于生成了两个空间权重表示: (B,C,H,W)-->(B,2,H,W)
    attention_weights = F.softmax(multi, dim=1) # 执行softmax归一化: (B,2,H,W)
    A, B = attention_weights.split(1, dim=1)  # 分割对对应于x和g的两个权重: A = B = (B,1,H,W)
    x_att = A.expand_as(x_) * x_  # A扩展到C个通道: (B,1,H,W)-expand->(B,C,H,W), 然后与原始的输入x_对应点相乘, 进一步调整
    g_att = B.expand_as(g_) * g_  # g扩展到C个通道: (B,1,H,W)-expand->(B,C,H,W), 然后与原始的输入g_对应点相乘, 进一步调整
    x_att = x_att + x_ # 添加残差连接
    g_att = g_att + g_ # 添加残差连接

    # 3、空间交互和交叉调制
    x_sig = torch.sigmoid(x_att) # 为x生成权重
    g_att_2 = x_sig * g_att # x调整g
    g_sig = torch.sigmoid(g_att) # 为g生成权重
    x_att_2 = g_sig * x_att # g调整x
    interaction = x_att_2 * g_att_2 # 执行逐元素乘法使融合后的特征更稳定

    # 4、重新校准
    projected = torch.sigmoid(self.bn(self.proj(interaction))) # 通过1×1Conv细化融合特征,然后通过sigmoid函数生成注意力权重
    weighted = projected * x_  #注意力权重重新校准初始输入x
    y = self.conv_block(weighted) # 通过1×1Conv生成输入
    y = self.bn_2(y)
    return y


if __name__ == '__main__':
    # (B,C,H,W)
    x1 = torch.randn(1, 64, 224,224)
    x2 = torch.randn(1, 64, 224, 224)

    Model = MultiScaleGatedAttn(dim=64)
    out = Model(x1,x2)
    print(out.shape)