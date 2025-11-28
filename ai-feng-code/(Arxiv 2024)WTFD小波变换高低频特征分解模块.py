import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
# https://github.com/yysdck/SFFNet/blob/main/models/FMS.py
# https://arxiv.org/pdf/2405.01992

'''
即插即用模块：WTFD 小波变换高低频特征分解模块 
WTFD 原理：
WTFD 模块利用 Haar 小波变换，将图像的空间特征分解为低频和高频成分。
这些成分分别捕获了图像中的低频和高频信息。低频特征包含图像的整体结构和大尺度连续区域的信息
，而高频特征则捕捉到边缘、纹理等局部细节。
Haar 小波是一种多分辨率分解方法，能够在不同尺度下提取图像的局部和全局信息。
通过对输入图像进行两次 Haar 变换，WTFD 得到四种频域特征：
低频近似系数（A）、水平高频系数（H）、垂直高频系数（V）和对角高频系数（D）。
这些高频信息代表图像中的细节部分，而低频则捕捉整体结构。

WTFD 作用：
通过引入频域特征，WTFD 有效地补充了空间特征无法捕捉的图像区域（如阴影和高频纹理区域）。
这对处理灰度变化显著的区域（例如阴影、纹理边缘等）具有重要意义。
高频特征帮助提高模型在边缘和细节区域的分割精度，而低频特征则增强模型对大尺度和阴影区域的理解能力。

适用于：遥感语义分割，图像分割，目标检测等所有CV任务
'''

class WTFD(nn.Module): #小波变化高低频分解模块
    def __init__(self, in_ch, out_ch):
        super(WTFD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)
        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)
        return yL,yH


class WTFDown(nn.Module):#小波变化高低频分解下采样模块
    def __init__(self, in_ch, out_ch):
        super(WTFDown, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)
        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)
        # return yL , yH
        return yL + yH #小波变化高低频分解下采样模块

if __name__ == "__main__":
    # 创建一个简单的输入特征图
    input = torch.randn(1,32, 64, 64)
    # 创建一个 WTFD实例
    WTFD =  WTFD(32,64)
    # 将输入特征图传递给 WTFD模块
    output_L,output_H = WTFD(input) #小波变化高低频分解模块
    # 打印输入和输出的尺寸
    print(f"input  shape: {input.shape}")
    print(f"output_L shape: {output_L.shape}")
    print(f"output_H shape: {output_H.shape}")

    print('-----------')
    WTFDown = WTFDown(32,64)  #小波变化高低频分解下采样模块
    output = WTFDown(input)
    print(f"input  shape: {input.shape}")
    print(f"output shape: {output.shape}")



