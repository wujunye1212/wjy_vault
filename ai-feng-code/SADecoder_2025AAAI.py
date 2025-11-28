import torch
import torch.nn as nn

__all__= ['SADecoder']
'''
来自AAAI 2025顶会论文    解码器模块 可以说所有计算机视觉任务都需要的模块         
即插即用模块： SAD 尺寸感知解码器模块 （也可以称为：多尺度解码器模块）
                  解码器创新也是CVPR一个热点
                  
我们设计了一个简单但有效的尺寸感知解码器（sa-解码器）。
sa-解码器通过在不同的层中分配不同大小的实体来实现单独的预测。
在多个cdfa输出的特征中，浅层特征图包含更多细粒度的信息，适合于预测较小规模的实体。
随着图层的加深，特征图包含了越来越多的全局信息和更高级的语义，使它们更适合于预测更大规模的实体。
因此，我们建立了三个小、中、大尺寸的解码器：每个解码器都接收来自相邻两个CDFAs的特征，
分别是Fe1和Fe2、Fe2和Fe3、Fe3和Fe4。然后，这三个解码器的输出沿着通道维度进行连接和融合。
然后通过s型函数生成预测的掩模。通过多个并行sa-解码器的协同工作，能够准确区分不同大小的个体。

所有计算机视觉任务通用的即插即用模块!
'''
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x
class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x  # [B,C,H,W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)
class SADecoder(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.c1 = CBR(in_c + out_c, out_c, kernel_size=1, padding=0)
        self.c2 = CBR(out_c, out_c, act=False)
        self.c3 = CBR(out_c, out_c, act=False)
        self.c4 = CBR(out_c, out_c, kernel_size=1, padding=0, act=False)
        self.ca = channel_attention(out_c)
        self.sa = spatial_attention()

    def forward(self,x, x2):
        x = self.up(x) #x 小尺寸的，x2是大尺寸的
        x = torch.cat([x, x2], axis=1)
        x = self.c1(x)
        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)
        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)
        s3 = x
        x = self.c4(x)
        x = self.relu(x + s3 + s2 + s1)
        x = self.ca(x)
        x = self.sa(x)
        return x
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # 实例化 SADecoder模块
    sad = SADecoder(in_c=128,out_c=64)
    x1 = torch.randn(1,128,32,32)
    x2 = torch.randn(1, 64, 64, 64)
    output = sad(x1,x2)
    # 打印输出张量的形状
    print("Input x1 shape:", x1.shape)
    print("Input x2 shape:", x2.shape)
    print("Output shape:", output.shape)