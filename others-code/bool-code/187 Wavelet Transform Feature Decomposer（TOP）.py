import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward  # 用于实现离散小波变换（DWT）
# pip install pywavelets==1.7.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
"""
    论文地址：https://arxiv.org/pdf/2405.01992
    论文题目：SFFNet: A Wavelet-Based Spatial and Frequency Domain Fusion Network for Remote Sensing Segmentation (TOP期刊)
    中文题目：SFFNet：一种基于小波的空间域与频域融合网络，用于遥感图像分割
    讲解视频：https://www.bilibili.com/video/BV1KZsyzUEVZ/
    小波变换特征分解器（Wavelet Transform Feature Decomposer, WTFD）：
        实际意义：①灰度变化敏感区域分割困难：在阴影、边缘、纹理突变处，空间域特征易受光照与背景变化影响，导致边界模糊、类别混淆。
                ②仅用频域特征会丢失空间结构信息：小波变换/傅里叶方法虽然能捕获灰度突变特征，但会破坏空间位置信息，导致目标形状或语义不完整。
        实现方式：先对输入空间域特征应用 Haar 小波变换，将其分解为 1 个低频分量（A，对应图像全局平滑信息）和 3 个高频分量（水平、垂直、
                对角方向的边缘与细节信息），为空间域特征融合提供频域信息支撑。
"""

class WTFD(nn.Module):
    """
    WTFD：Wavelet Transform Feature Decomposition（小波变换特征分解模块）
    将输入特征分解为低频（平滑）与高频（边缘/细节）并分别卷积处理。
    """
    def __init__(self, in_ch, out_ch):
        super(WTFD, self).__init__()
        # 1 层 Haar 小波分解
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        # 高频分支：HL/LH/HH 拼成 3*in_ch 通道 -> 1x1 降回 in_ch
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 3, in_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

        # 低频输出通道对齐
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # 高频输出通道对齐
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """ 前向传播逻辑：
            输入 x -> 小波分解 -> 高频拼接 -> 卷积融合 -> 输出低频与高频特征
         """
        # 使用小波变换进行特征分解 # yL: 低频部分（平滑信息）
        # yH: 高频部分（包含3个方向的细节信息）
        yL, yH = self.wt(x)

        # 从高频系数中提取3个方向分量：HL（水平方向）、LH（垂直方向）、HH（对角方向）
        yH0 = yH[0]                 # [B, C, 3, H2, W2]
        y_HL = yH0[:, :, 0, :, :]   # [B, C, H2, W2]
        y_LH = yH0[:, :, 1, :, :]
        y_HH = yH0[:, :, 2, :, :]
        # # 沿通道维拼接三个方向的高频信息
        yH_cat = torch.cat([y_HL, y_LH, y_HH], dim=1)  # [B, 3*C, H2, W2]
        # 对拼接后的高频特征进行卷积 + BN + ReLU
        yH_feat = self.conv_bn_relu(yH_cat)            # [B, C, H2, W2]

        # 分别对低频与高频特征进行输出卷积处理
        yL_out = self.outconv_bn_relu_L(yL)           # [B, out_ch, H2, W2]
        yH_out = self.outconv_bn_relu_H(yH_feat)      # [B, out_ch, H2, W2]

        return yL_out, yH_out

if __name__ == "__main__":
    # 输入的 H、W 建议为偶数，以适配 1 层小波下采样
    x = torch.randn(1, 32, 50, 50)
    model = WTFD(32, 32)
    out_L, out_H = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"低频输出张量形状: {out_L.shape}")
    print(f"高频输出张量形状: {out_H.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
