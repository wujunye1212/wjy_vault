import math
import torch
from torch import nn
# https://github.com/Lose-Code/UBRFC-Net
# https://www.sciencedirect.com/science/article/abs/pii/S0893608024002387
'''
 用于图像去雾的无监督双向对比重建和自适应细粒度信道注意力网络     SCI 一区 2024 顶刊
捕捉全局和局部信息交互即插即用注意力模块：FCAttention

无监督算法在图像去雾领域取得了显著成果。此外，SE通道注意力机制仅利用全连接层捕捉全局信息，
缺乏与局部信息的互动，导致图像去雾时特征权重分配不准确。

为克服这些挑战，我们开发了一种自适应细粒度通道注意力（FCA）机制，
利用相关矩阵在不同粒度级别捕获全局和局部信息之间的关联，促进了它们之间的互动，实现了更有效的特征权重分配。

在图像去雾方面超越了当前先进的方法。本研究成功地引入了一种增强型无监督图像去雾方法，有效解决了现有技术的局限，实现了更优的去雾效果。
适用于：图像增强，暗光增强，图像去雾，图像去噪等所有CV2维任务通用的即插即用注意力模块
'''
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class FCAttention(nn.Module):
    def __init__(self,channel,b=1, gamma=2):
        super(FCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        #一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()


    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)#(1,64,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)#(1,1,64)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)#(1,64,1,1)
        #x1 = x1.transpose(-1, -2).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)

        #out2 = self.fc(x)
        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return input*out

# 适用于：图像增强，暗光增强，图像去雾，图像去噪
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1,64,256,256)
    model = FCAttention(channel=64)
    output = model (input)
    print('input_size:', input.size())
    print('output_size:', output.size())
