import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
import time
import sys


class ShuffleAttn(nn.Module):
    def __init__(self, in_features, out_features, group=4, act_layer=nn.GELU):
        super().__init__()
        self.group = group
        self.in_features = in_features
        self.out_features = out_features

        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, out_features, groups=self.group, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size() # (B,C,H,W)
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group # d = C//G

        # 打乱通道
        x = x.reshape(batchsize, group_channels, self.group, height, width) # (B,C,H,W)--reshape-->(B,d,G,H,W)
        x = x.permute(0, 2, 1, 3, 4) # (B,d,G,H,W)-->(B,G,d,H,W)
        x = x.reshape(batchsize, num_channels, height, width) # (B,G,d,H,W)-reshape->(B,C,H,W)

        return x

    def channel_rearrange(self ,x):
        batchsize, num_channels, height, width = x.data.size() # (B,C,1,1)
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group # d = C//G

        x = x.reshape(batchsize, self.group, group_channels, height, width) # (B,C,1,1)--reshape-->(B,G,d,1,1)
        x = x.permute(0, 2, 1, 3, 4) # (B,G,d,1,1)-->(B,d,G,1,1)
        x = x.reshape(batchsize, num_channels, height, width) # (B,d,G,1,1)--reshape-->(B,C,1,1)

        return x

    def forward(self, x):
        res = x
        x = self.channel_shuffle(x) # 打乱通道排序: (B,C,H,W)-->(B,C,H,W)
        x = self.gating(x) # (B,C,H,W)-->(B,C,1,1)
        x = self.channel_rearrange(x) # 恢复原有通道顺序: (B,C,1,1)
        y = x * res  # 调整初始输入值
        return y


if __name__ == '__main__':
    # (B,C,H,W)
    x1 = torch.randn(1, 64, 224, 224)

    # 定义ShuffleAttn
    Model = ShuffleAttn(in_features=64, out_features=64, group=16)
    # 执行ShuffleAttn
    out = Model(x1)
    print(out.shape)