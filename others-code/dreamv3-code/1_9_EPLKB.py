import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CA(nn.Module):
    def __init__(self, channels):
        super(CA, self).__init__()
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.AdaptiveAvgPool(x)) # 生成权重: (B,d,H,W)--pool-->(B,d,1,1)--sigmoid-->(B,d,1,1)
        out = out * x # 使用通道权重调整每个通道的重要性: (B,d,1,1) * (B,d,H,W) == (B,d,H,W)
        return out



class PLKB(nn.Module):
    '''
    corresponding to Enhanced Partial Large Kernel Block (EPLKB) in paper
    '''

    def __init__(self, channels, large_kernel, split_group):
        super(PLKB, self).__init__()
        self.channels = channels
        self.split_group = split_group
        self.split_channels = int(channels // split_group)
        self.CA = CA(channels)
        self.DWConv_Kx1 = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=(large_kernel, 1), stride=1,
                                    padding=(large_kernel // 2, 0), groups=self.split_channels)
        self.DWConv_1xK = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=(1, large_kernel), stride=1,
                                    padding=(0, large_kernel // 2), groups=self.split_channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        # channel shuffle
        B, C, H, W = x.size()
        x = x.reshape(B, self.split_channels, self.split_group, H, W) # (B,C,H,W)-->(B,d,N,H,W);  C=d*N,N是组的个数,d是每组的通道数
        x = x.permute(0, 2, 1, 3, 4) # (B,d,N,H,W)-->(B,N,d,H,W), 改变d和N的顺序
        x = x.reshape(B, C, H, W) # (B,N,d,H,W)-->(B,C,H,W), 这样的操作会改变通道的顺序

        x1, x2 = torch.split(x, (self.split_channels, self.channels - self.split_channels), dim=1) # 从中抽出一部分要进行增强的通道: (B,C,H,W)--> x1:(B,d,H,W), x2:(B,C-d,H,W)

        # channel attention
        x1 = self.CA(x1) # 首先使用通道注意力增强通道表示: (B,d,H,W)-->(B,d,H,W)

        x1 = self.DWConv_Kx1(self.DWConv_1xK(x1)) # 使用垂直和水平条状大核卷积进行空间信息提取: (B,d,H,W)-->(B,d,H,W)
        out = torch.cat((x1, x2), dim=1) # 重新拼接"被增强特征“ 与 ”旁路通道特征“：(B,d,H,W)--cat--(B,C-d,H,W) == (B,C,H,W)
        out = self.act(self.conv1(out)) # 1×1Conv进行通道融合: (B,C,H,W)-->(B,C,H,W)
        return out


if __name__ == '__main__':

    # (B,C,H,W)
    x1 = torch.randn(1, 64, 224, 224)
    B,C,H,W = x1.size()

    # 定义 PLKB
    Model = PLKB(channels=C, large_kernel=31, split_group=4)

    # 执行 PLKB
    out = Model(x1)
    print(out.shape)