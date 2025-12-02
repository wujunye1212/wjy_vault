import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
# 按照这个第三方库需要安装pip install pytorch_wavelets==1.3.0
# 如果提示缺少pywt库则安装 pip install PyWavelets



'''
# 中科院一区顶刊 2023
即插即用模块：HWD小波下采样模块

Haar Wavelet Downsampling (HWD)模块是一种用于语义分割任务的简便而有效的下采样模块。
该模块通过Haar小波变换减少特征图的空间分辨率，同时尽可能多地保留信息，进而提升语义分割模型的性能。
HWD模块可以轻松集成到卷积神经网络（CNNs）中，替代传统的下采样方法如最大池化或步幅卷积。

HWD小波下采样模块简单介绍：
1.模块组成：HWD模块由两个主要部分组成：无损特征编码块和特征表示学习块。
第一个部分通过Haar小波变换有效地减少特征图的空间分辨率，同时保留所有信息。
第二个部分则通过标准的卷积层、批归一化和ReLU激活函数来过滤冗余信息，以便后续层更有效地学习代表性特征。

2.模块优势：传统的下采样方法通常会导致关键信息（如边界、纹理等）的丢失，这在语义分割任务中特别不利。
而HWD模块通过将部分空间信息编码到通道维度中，从而在减少分辨率的同时最大限度地保留了原始信息。

3.性能表现：实验证明，HWD模块在多种CNN架构中都能显著提高语义分割的性能，
尤其是在小尺度物体的分割中表现更为优异。此外，HWD模块能够减少特征图的不确定性
，从而帮助网络生成更加确定的预测结果。

4.通用性：HWD模块可以直接替换掉现有的下采样层，如步幅卷积或池化层，而不会显著增加计算开销。
'''

class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = HWD(in_ch=32, out_ch=64)  # 输入通道数，输出通道数
    input = torch.rand(1, 32, 64, 64)
    output = block(input)
    print('input :',input.size())
    print('output :', output.size())
