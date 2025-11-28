import torch.nn as nn
import torch


class MBRConv1(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale

        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)

    def forward(self, inp):

        # 以右分支为例进行注释
        x0 = self.conv(inp) # (B,1,H,W)--1×1conv-->(B,4*C,H,W)
        x = torch.cat([x0, self.conv_bn(x0)], 1) # (B,4*C,H,W)--cat--(B,4*C,H,W)==(B,8*C,H,W)
        out = self.conv_out(x) # (B,8*C,H,W)-->(B,C,H,W)
        return out



class HDPA(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(HDPA, self).__init__()
        self.channels = channels

        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(channels, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.att1= nn.Sequential(
            MBRConv1(1, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )


    def forward(self, x1):
        x2 = self.att(x1) # 左分支(avgpool->MBRConv->Sigmoid): (B,C,H,W)--avgpool-->(B,C,1,1)--MBRConv1-->(B,C,1,1)
        max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True) # 元素点积-->maxpool: step1:(B,C,1,1) * (B,C,H,W) == (B,C,H,W); step2 通道最大池化:(B,C,H,W)--maxpool-->(B,1,H,W)
        x3 = self.att1(max_out)  # 右分支(MBRConv->Sigmoid): (B,1,H,W)-->(B,C,H,W)
        x4 = torch.mul(x3, x2) * x1 # 权重调整最初的输入: (B,C,H,W)-mul-(B,C,1,1)==(B,C,H,W); (B,C,H,W)*(B,C,H,W)==(B,C,H,W)
        return x4



if __name__ == '__main__':
    # (B,C,H,W)
    x1 = torch.randn(8, 64, 32, 32)

    # 定义HDPA
    Model = HDPA(channels=64)
    # 执行HDPA
    out = Model(x1)
    print(out.shape)