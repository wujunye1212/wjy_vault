import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import DropPath, trunc_normal_
from typing import List
from typing import Tuple
import sys
import os
from timm.models.layers import DropPath, Mlp, to_2tuple
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        dim1 = dim
        dim = dim // 2

        self.dwconv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.dwconv3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv4 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.dwconv5 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv6 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(dim1)

        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(dim1)

        self.act3 = nn.GELU()
        self.bn3 = nn.BatchNorm2d(dim1)

    def forward(self, x, H, W):
        B, N, C = x.shape  # (B,N,C1) 这里的C1是降维后的通道,后面用C指代
        n = N // 21  # 1029//21=49

        # 将flatten的像素点重塑为2D特征图,便于后续进行卷积操作
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()  # (B,16*n,C)=(B,784,C)-transpose->(B,C,784)-view->(B,C,28,28)
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()   # (B,4*n,C)=(B,196,C)-transpose->(B,C,196)-view->(B,C,14,14)
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous() # (B,20*n:,C)=(B,49,C)-transpose->(B,C,49)-view->(B,C,7,7)

        # 将第一个尺度特征X1在通道上进行分割, 然后应用不同感受野的卷积层
        x11, x12 = x1[:, :C // 2, :, :], x1[:, C // 2:, :, :] # (B,C,28,28)-split-> (B,C/2,28,28) and (B,C/2,28,28)
        x11 = self.dwconv1(x11)  # 第一组应用3×3Conv: (B,C/2,28,28)-->(B,C/2,28,28)
        x12 = self.dwconv2(x12)  # 第二组应用5×5Conv: (B,C/2,28,28)-->(B,C/2,28,28)
        x1 = torch.cat([x11, x12], dim=1) # 在通道上拼接第一组和第二组: (B,C,28,28)
        x1 = self.act1(self.bn1(x1)).flatten(2).transpose(1, 2) # flatten,便于恢复和输入相同的shape: (B,C,28,28)-->(B,C,784)-->(B,784,C)

        # 将第二个尺度特征X2在通道上进行分割, 然后应用不同感受野的卷积层
        x21, x22 = x2[:, :C // 2, :, :], x2[:, C // 2:, :, :]   # (B,C,14,14)-split-> (B,C/2,14,14) and (B,C/2,14,14)
        x21 = self.dwconv3(x21) # 第一组应用3×3Conv: (B,C/2,14,14)-->(B,C/2,14,14)
        x22 = self.dwconv4(x22) # 第二组应用5×5Conv: (B,C/2,14,14)-->(B,C/2,14,14)
        x2 = torch.cat([x21, x22], dim=1) # 在通道上拼接第一组和第二组: (B,C,14,14)
        x2 = self.act2(self.bn2(x2)).flatten(2).transpose(1, 2) # flatten,便于恢复和输入相同的shape: (B,C,14,14)-->(B,C,196)-->(B,196,C)

        # 将第三个尺度特征X3在通道上进行分割, 然后应用不同感受野的卷积层
        x31, x32 = x3[:, :C // 2, :, :], x3[:, C // 2:, :, :] # (B,C,7,7)-split-> (B,C/2,7,7) and (B,C/2,7,7)
        x31 = self.dwconv5(x31) # 第一组应用3×3Conv: (B,C/2,7,7)-->(B,C/2,7,7)
        x32 = self.dwconv6(x32) # 第二组应用5×5Conv: (B,C/2,7,7)-->(B,C/2,7,7)
        x3 = torch.cat([x31, x32], dim=1) # 在通道上拼接第一组和第二组: (B,C,7,7)
        x3 = self.act3(self.bn3(x3)).flatten(2).transpose(1, 2)  # flatten,便于恢复和输入相同的shape: (B,C,7,7)-->(B,C,49)-->(B,49,C)

        x = torch.cat([x1, x2, x3], dim=1) # 拼接三个尺度的特征,恢复和输入相同的shape: (B,1029,C)
        return x


class MRFP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MultiDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x) # (B,N,C)-->(B,N,C1) C1是降维后的通道数。  N=(HW/8^2) + (HW/16^2) + (HW/32^2)
        x = self.dwconv(x, H, W) # (B,N,C1)-->(B,N,C1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x) # (B,N,C1)-->(B,N,C)
        x = self.drop(x)
        return x


if __name__ == '__main__':
    # (B,N,C)   N表示像素的个数
    # C1/C2/C3中的784/196/49不可更改, C1是在原始特征图224×224进行卷积得到的,C2是在C1基础上卷积得到的,C3是在C2基础上卷积得到的
    # 如果想直接利用论文中设置的卷积层得到这三个不同尺度的特征,可以查看 ”6_10_ViTCoMer_补充“ 文件
    c1 = torch.randn(1,784,64)  # HW/8^2 = 784
    c2 = torch.randn(1,196,64)  # HW/16^2 = 196
    c3 = torch.randn(1,49,64)  # HW/32^2 = 49
    c = torch.cat([c1, c2, c3], dim=1) # 将特征图flatten,然后拼接: (B,N,C)=(1,1029,64)  N=(HW/8^2) + (HW/16^2) + (HW/32^2)
    _, _, dim = c.shape

    Model = MRFP(in_features=dim, hidden_features=int(dim * 0.5))
    out = Model(c,H=14,W=14)  # H:每列有多少个patch; W: 每行有多少个Patch. 默认的patch size == 16
    print(out.shape)