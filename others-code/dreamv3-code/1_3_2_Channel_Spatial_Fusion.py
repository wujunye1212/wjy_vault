import numpy as np
import torch
from torch import nn
from torch.nn import init

"CBAM: Convolutional Block Attention Module "


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)  # 通过最大池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1)
        avg_result = self.avgpool(x)  # 通过平均池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1)
        max_out = self.se(max_result)  # 共享同一个MLP: (B,C,1,1)--> (B,C,1,1)
        avg_out = self.se(avg_result)  # 共享同一个MLP: (B,C,1,1)--> (B,C,1,1)
        output = max_out + avg_out  # (B,C,1,1)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:(B,C,H,W)
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 最大值和对应的索引.
        avg_result = torch.mean(x, dim=1, keepdim=True)  # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 平均值
        result = torch.cat([max_result, avg_result], 1)  # 在通道上拼接两个矩阵:(B,2,H,W)
        output = self.conv(result)  # 然后重新降维为1维:(B,1,H,W); 在这里并没有按照模型里的的方式先MLP,然后再Add; 而是先concat,再Conv; 实际含义是一致的,就是实现方式不一致。
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49, HW=None):
        super().__init__()
        self.ChannelAttention = ChannelAttention(channel=channel, reduction=reduction)
        self.SpatialAttention = SpatialAttention(kernel_size=kernel_size)
        self.joint_channel = channel + HW
        self.MLP = nn.Sequential(
            nn.Conv2d(self.joint_channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, self.joint_channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # (B,C,H,W)
        B, C, H, W = x.size()
        residual = x
        Channel_x = self.ChannelAttention(x).reshape(B,C,1,1) # (B,C,1,1)-->(B,C,1,1)
        Spatial_x = self.SpatialAttention(x).reshape(B,H*W,1,1) # (B,1,H,W)-->(B,HW,1,1)

        # 拼接,然后通过MLP建立相关性
        CS_x = torch.cat([Channel_x, Spatial_x], 1) # (B,C,1,1)-Conca->(B,HW,1,1)-->(B,C+HW,1,1)
        CS_xx = self.MLP(CS_x) # (B,C+HW,1,1)-降维->(B,M,1,1)-升维->(B,C+HW,1,1)

        # 拆分,然后通过sigmoid得到权重表示
        Channel_x = CS_xx[:,:C,:].reshape(B,C,1,1)  # (B,C,1,1)-->(B,C,1,1)
        Spatial_x = CS_xx[:,C:,:].reshape(B,1,H,W) # (B,HW,1,1)-->(B,1,H,W)
        Channel_weight = self.sigmoid(Channel_x)
        Spatial_weight = self.sigmoid(Spatial_x)

        # 分别得到通道和空间权重之后,既可以单独相乘得到两个输出, 也可以一块与X相乘得到一个输出,视自己的任务来定义
        out1 = x * Channel_weight  # 将输入与通道注意力权重相乘: (B,C,H,W) * (B,C,1,1) = (B,C,H,W)
        out2 = x * Spatial_weight  # 将更新后的输入与空间注意力权重相乘:(B,C,H,W) * (B,1,H,W) = (B,C,H,W)
        return out1,out2


if __name__ == '__main__':
    # (B,C,H,W)  注意: 因为在模型中需要将HW和C拼接起来,所在在输入到模型的时候,最好把通道C和HW做个降维(池化、下采样均可),然后在输入到模型中去,输出之后再恢复shape就可以了！
    input = torch.randn(1, 64, 7, 7)
    B,C,H,W=input.shape
    Model = CBAMBlock(channel=64, reduction=8, kernel_size=7, HW=H*W)
    out1,out2 = Model(input)
    print(out1.shape,out2.shape)

