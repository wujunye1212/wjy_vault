import torch
import torch.nn as nn
import torch.nn.functional as F
import math
'''
 ASF-YOLO：一种具有注意力尺度序列融合的新型 YOLO 模型，用于细胞实例分割  
 
多尺度特征提取即插即用模块：SSFF       2024 SCI

针对细胞图像的多尺度问题，现有文献中采用特征金字塔结构进行特征融合，其中仅采用求和或串联来融合金字塔特征。
然而，各种特征金字塔网络的结构并不能有效地利用所有金字塔特征图之间的相关性。
本文提出了一种新型的SSFF模块，该模块能够更好地将多尺度特征图（即深层特征图的高级特征信息）
与具有相同长宽比的浅层特征图的详细特征信息相结合。   简单理解SSFF：高频特征与低频特征多尺度融合模块
我们进一步构建了从主干网（即P3、P4和P5）生成的多尺度特征图的顺序表示，
这些特征图以不同的细节或比例层次捕获图像内容。

实验表明， 在 2018 Data Science Bowl 数据集上表现优异，
实现了 0.91 的检测 mAP、0.887 的掩码 mAP 和 47.3 FPS 的推理速度。

适用于：小目标分割，小目标检测等所有CV2维任务通用多尺度特征提取模块SSFF
'''
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Zoom_cat(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv_l_post_down = Conv(in_dim, 2*in_dim, 3, 1, 1)

    def forward(self, x):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        lms = torch.cat([l, m, s], dim=1)
        return lms
class SSFF(nn.Module):
    def __init__(self, inc, channel):
        super(SSFF, self).__init__()
        self.conv0 = Conv(inc[0], channel, 1)
        self.conv1 = Conv(inc[1], channel, 1)
        self.conv2 = Conv(inc[2], channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]
        p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x

# SSFF 多尺度特征提取模块
if __name__ == '__main__':
    # 模型对象实例化
    model = SSFF(inc =[64, 128, 256], channel=512)
    # 模拟输入数据
    input_data = [
        torch.randn(1, 64, 64, 64), # l特征图 P3
        torch.randn(1, 128, 32, 32),  # m特征图 P4
        torch.randn(1, 256, 16, 16)   # s特征图 P5
    ]
    # 前向传播
    output = model(input_data)
    print('Output size:', output.size())   #输出特征图的H，W大小==l的尺寸，


