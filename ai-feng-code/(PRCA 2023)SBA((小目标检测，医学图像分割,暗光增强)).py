import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init
warnings.filterwarnings('ignore')

'''
DuAT：用于医学图像分割的双聚合 Transformer 网络
SBA：选择性边界融合模块

本文在参考文献中观察到，浅层和深层的特征是互补的。浅层层的语义较少，但细节丰富，
边界更明显，失真更少。此外，该深层还包含了丰富的语义信息。
因此，直接将低级特征与高级特征融合可能会导致冗余和不一致性。

为了解决这个问题，我们提出了SBA模块，它有选择地聚合边界信息和语义信息，
以描述更细粒度的对象轮廓和重新校准对象的位置。

与以往的融合方法不同，我们设计了一种新的重新校准注意单元（RAU）块，
它在融合前自适应地从两个输入（F s，Fb）中获取相互表示。
如图2所示，将浅层和深层信息通过不同的方式输入到两个RAU块中，
以弥补高级语义特征缺失的空间边界信息和低层特征缺失的语义信息缺失。
最后，两个RAU块的输出在3×3卷积后被连接起来。
该聚合策略实现了不同特征的鲁棒组合，并对粗糙特征进行了细化。

'''
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class Block(nn.Sequential):
    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True, norm_layer=nn.BatchNorm2d):
        super(Block, self).__init__()
        if bn_start:
            self.add_module('norm1', norm_layer(input_num)),

        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', norm_layer(num1)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                           dilation=dilation_rate, padding=dilation_rate)),
        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(Block, self).forward(_input)
        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)
        return feature
def Upsample(x, size, align_corners=False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)
def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)
class SBA(nn.Module):

    def __init__(self, input_dim=64,output_dim = 64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = BasicConv2d(input_dim // 2, input_dim // 2, 1)
        self.d_in2 = BasicConv2d(input_dim // 2, input_dim // 2, 1)

        self.conv = nn.Sequential(BasicConv2d(input_dim, input_dim, 3, 1, 1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))
        self.fc1 = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, bias=False)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, H_feature, L_feature):
        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)

        g_L_feature = self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)

        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(g_H_feature * H_feature,
                                                                                       size=L_feature.size()[2:],
                                                                                       align_corners=False)
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(g_L_feature * L_feature,
                                                                                       size=H_feature.size()[2:],
                                                                                       align_corners=False)

        H_feature = Upsample(H_feature, size=L_feature.size()[2:])
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out

if __name__ == '__main__':
    input1 = torch.randn(1, 32, 64, 64) # x: (B, C,H, W)
    input2 = torch.randn(1, 32, 64, 64) # x: (B, C,H, W)
    model = SBA(input_dim=32,output_dim=32)
    output = model(input1,input2)
    print("SBA_input size:", input1.size())
    print("SBA_Output size:", output.size())