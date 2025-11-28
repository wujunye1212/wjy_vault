import torch
import torch.nn as nn
'''
    论文地址：https://openaccess.thecvf.com/content/CVPR2024/html/Cong_A_Semi-supervised_Nighttime_Dehazing_Baseline_with_Spatial-Frequency_Aware_and_Realistic_CVPR_2024_paper.html
    论文题目：A Semi-supervised Nighttime Dehazing Baseline with Spatial-Frequency Aware and Realistic Brightness Constraint（CVPR 2024）
    中文题目：具有空间频率感知和现实亮度约束的半监督夜间除雾基线模型
    讲解视频：https://www.bilibili.com/video/BV1CE1uYDEf4/
      频率域动态聚合（Frequency Spectrum Dynamic Aggregation,FSDA）：
            通过傅里叶变换将图像转换到频域，并在频域中进行动态滤波处理，以提取不同通道中的频率特征。
'''

# 定义SE（Squeeze-and-Excitation）层
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 线性变换降维
            nn.ReLU(inplace=True),  # ReLU激活
            nn.Linear(channel // reduction, channel, bias=False),  # 线性变换升维
            nn.Sigmoid()  # Sigmoid激活
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入尺寸
        y = self.avg_pool(x).view(b, c)  # 全局平均池化后展平
        y = self.fc(y).view(b, c, 1, 1)  # 经过线性变换并恢复形状
        return x * y.expand_as(x)  # 对输入进行加权

# 定义频谱动态聚合模块
"""
   例如在一幅夜间有雾且存在光晕和噪声的图像中，雾霾部分可能在低频区域表现明显，光晕可能在某些特定频率范围影响图像，噪声则分布在高频区域。
   1、能够根据频率特性，动态调整滤波器参数，准确地识别和处理各个部分的频谱信息，将它们从原始图像频谱中分离出来。
   2、在计算通道权重图时，综合考虑各个通道频谱信息经过卷积、池化和激活函数后的结果，使得最终的通道权重能够反映出不同通道频谱信息的重要性。
   3、然后将这些权重应用于频谱聚合过程，使得各个通道的频谱信息能够按照其重要性进行重新组合，得到更能代表图像整体特征且包含丰富频率信息的频谱表示。 
"""
class Frequency_Spectrum_Dynamic_Aggregation(nn.Module):
    # 初始化函数
    def __init__(self, nc):
        super(Frequency_Spectrum_Dynamic_Aggregation, self).__init__()  # 调用父类初始化方法
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),  # 1x1卷积处理幅度
            nn.LeakyReLU(0.1, inplace=True),  # LeakyReLU激活
            SELayer(channel=nc),  # SE层  计算通道权重图 相乘
            nn.Conv2d(nc, nc, 1, 1, 0)  # 另一个1x1卷积
        )
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),  # 1x1卷积处理相位
            nn.LeakyReLU(0.1, inplace=True),  # LeakyReLU激活
            SELayer(channel=nc),  # SE层 计算通道权重图 相乘
            nn.Conv2d(nc, nc, 1, 1, 0)  # 另一个1x1卷积
        )

    # 前向传播函数
    def forward(self, x):
        _, _, H, W = x.shape  # 获取输入尺寸
        # input = torch.rand(4, 32, 64, 64)

        # norm='forward'，则会在前向变换时应用归一化因子
        # norm='backward'则是在逆变换（即irfft2）时应用归一化因子
        x_freq = torch.fft.rfft2(x, norm='backward')  # 计算二维实数FFT  torch.Size([4, 32, 64, 33])

        """
            1、对于不同通道的幅度谱和相位谱，通过点卷积操作进行聚合。
        """
        ori_mag = torch.abs(x_freq)  # 计算幅度 torch.Size([4, 32, 64, 33])
        mag = self.processmag(ori_mag)  # 处理幅度   torch.Size([4, 32, 64, 33])  这里计算通道权重图
        mag = ori_mag + mag  # torch.Size([4, 32, 64, 33])  针对性地增强或抑制特定频率成分，有效分离和处理雾霾、光晕和噪声的频率特征，从而为后续去除这些干扰做好准备。

        ori_pha = torch.angle(x_freq)  # 计算相位  torch.Size([4, 32, 64, 33])
        pha = self.processpha(ori_pha)  # 处理相位  torch.Size([4, 32, 64, 33]) 这里计算通道权重图
        pha = ori_pha + pha  # torch.Size([4, 32, 64, 33])  针对性地增强或抑制特定频率成分，有效分离和处理雾霾、光晕和噪声的频率特征，从而为后续去除这些干扰做好准备。


        """
            2、得到滤波后的频谱 对应的实部和虚部
        """
        real = mag * torch.cos(pha)  # 计算实部 torch.Size([4, 32, 64, 33])
        imag = mag * torch.sin(pha)  # 计算虚部 torch.Size([4, 32, 64, 33])
        x_out = torch.complex(real, imag)  # 合成复数输出  torch.Size([4, 32, 64, 33])

        """
            在频域进行动态参数学习后，将特征图重新映射回空间域。
        """
        x_freq_spatial = torch.fft.irfft2(x_out, s=(H, W), norm='backward')  # 逆FFT转换回空间域 torch.Size([4, 32, 64, 64])
        return x_freq_spatial  # 返回结果

# 主程序入口
if __name__ == '__main__':
    block = Frequency_Spectrum_Dynamic_Aggregation(32)  # 实例化模型
    input = torch.rand(4, 32, 64, 64)  # 创建随机输入张量
    output = block(input)  # 计算模型输出
    print(input.size())  # 打印输入张量尺寸
    print(output.size())  # 打印输出张量尺寸

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")