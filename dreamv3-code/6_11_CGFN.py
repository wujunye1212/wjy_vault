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
        out = self.sigmoid(self.AdaptiveAvgPool(x))
        out = out * x
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
        x = x.reshape(B, self.split_channels, self.split_group, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B, C, H, W)

        x1, x2 = torch.split(x, (self.split_channels, self.channels - self.split_channels), dim=1)

        # channel attention
        x1 = self.CA(x1)

        x1 = self.DWConv_Kx1(self.DWConv_1xK(x1))
        out = torch.cat((x1, x2), dim=1)
        out = self.act(self.conv1(out))
        return out



class Scaler(nn.Module):
    def __init__(self, channels, init_value=1e-5, requires_grad=True):
        super(Scaler, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones(1, channels, 1, 1),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class CGFN(nn.Module):
    '''
    Cross-Gate Feed-Forward Network (CGFN)
    '''
    def __init__(self, channels, large_kernel, split_group):
        super(CGFN, self).__init__()
        self.PLKB = PLKB(channels, large_kernel, split_group)
        self.DWConv_3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.conv1 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.scaler1 = Scaler(channels)
        self.scaler2 = Scaler(channels)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.PLKB(x) # 执行A模块
        x1_scaler = self.scaler1(x - x1) # 输入减去A模块输出, 得到差异图x1_scaler

        x2 = self.DWConv_3(x) # 执行B模块
        x2_scaler = self.scaler2(x - x2) # 输入减去B模块输出, 得到差异图x2_scaler

        x1 = x1 * x2_scaler # 差异图x2_scaler 调整 A模块的输出
        x2 = x2 * x1_scaler # 差异图x1_scaler 调整 B模块的输出

        out = self.act(self.conv1(torch.cat((x1, x2), dim=1))) # 通道拼接, 然后1×1Conv融合
        return out


if __name__ == '__main__':

    # (B,C,H,W)
    x1 = torch.randn(1, 64, 224, 224)
    B,C,H,W = x1.size()

    # 定义 CGFN
    Model = CGFN(channels=C, large_kernel=31, split_group=4)

    # 执行 CGFN
    out = Model(x1)
    print(out.shape)