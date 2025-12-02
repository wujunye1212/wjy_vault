import torch
from torch import nn
from einops.layers.torch import Rearrange


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True) # 在通道方向上执行平均池化: (B,C,H,W)-->(B,1,H,W)
        x_max, _ = torch.max(x, dim=1, keepdim=True) # 在通道方向上执行最大池化: (B,C,H,W)-->(B,1,H,W)
        x2 = torch.cat([x_avg, x_max], dim=1) # 拼接平均池化和最大池化特征: (B,2,H,W)
        sattn = self.sa(x2) # (B,2,H,W)-->(B,1,H,W)
        return sattn



class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x) # 在空间方向上执行平均池化：(B,C,H,W)-->(B,C,1,1)
        cattn = self.ca(x_gap) # 两层全连接建模通道相关性: (B,C,1,1)-->(B,C,1,1)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # (B,C,H,W)-->(B,C,1,H,W)
        pattn1 = pattn1.unsqueeze(dim=2)  # (B,C,H,W)-->(B,C,1,H,W)
        x2 = torch.cat([x, pattn1], dim=2)  # (B,C,2,H,W)
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2) # (B,C,1,H,W)-->(B,2C,H,W)
        pattn2 = self.pa2(x2) # (B,2C,H,W)-->(B,C,H,W)
        #pattn2 = self.sigmoid(pattn2)
        return pattn2

class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y  #(B,C,H,W)
        cattn = self.ca(initial) # 计算通道注意力权重: (B,C,H,W)-->(B,C,1,1)
        sattn = self.sa(initial) # 计算空间注意力权重: (B,C,H,W)-->(B,1,H,W)
        pattn1 = sattn + cattn # 融合空间注意力权重和通道注意力权重: (B,C,H,W)
        pattn2 = self.sigmoid(self.pa(initial, pattn1)) # 得到CGA模块生成的权重: (B,C,H,W)-->(B,C,H,W)
        result = initial + pattn2 * x + (1 - pattn2) * y # 对深层特征和浅层特征进行加权, 并添加残差连接
        result = self.conv(result) # (B,C,H,W)
        return result


if __name__ == '__main__':
    # (B,C,H,W)
    x = torch.randn(1, 64, 224, 224)
    y = torch.randn(1, 64, 224, 224)

    Model = CGAFusion(dim=64)
    out = Model(x,y)
    print(out.shape)