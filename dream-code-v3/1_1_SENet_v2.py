import numpy as np
import torch
from torch import nn
from torch.nn import init

"Squeeze-and-Excitation Networks"


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        # 在空间维度上,将H×W压缩为1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 包含两层全连接,先降维,后升维。最后接一个sigmoid函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, 3*channel, bias=False),
        )


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

    def forward(self, x1,x2,x3):
        # (B,C,H,W)
        B, C, H, W = x1.size()
        x = x1 + x2 + x3
        # Squeeze: (B,C,H,W)-->avg_pool-->(B,C,1,1)-->view-->(B,C)
        y = self.avg_pool(x).view(B, C)
        # Excitation: (B,C)-->fc-->(B,3C)-->(B, 3C, 1, 1)
        y = self.fc(y).view(B, 3*C, 1, 1)
        # split
        weight1 = torch.sigmoid(y[:,:C,:,:]) # (B,C,1,1)
        weight2 = torch.sigmoid(y[:, C:2*C, :, :]) # (B,C,1,1)
        weight3 = torch.sigmoid(y[:, 2*C:, :, :]) # (B,C,1,1)
        # scale: (B,C,H,W) * (B,C,1,1) == (B,C,H,W)
        out = x1 * weight1 + x2 * weight2 + x3 * weight3
        return out


if __name__ == '__main__':
    # (B,C,H,W)
    input1 = torch.randn(1, 512, 7, 7)
    input2 = torch.randn(1, 512, 7, 7)
    input3 = torch.randn(1, 512, 7, 7)
    # 定义通道注意力
    Model = SEAttention(channel=512, reduction=8)
    output = Model(input1,input2,input3)
    print(output.shape)

