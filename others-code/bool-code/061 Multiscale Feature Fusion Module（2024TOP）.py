import torch
import torch.nn as nn

'''
    论文地址：https://ieeexplore.ieee.org/abstract/document/10504297
    论文题目：DGMA2-Net: A Difference-Guided Multiscale Aggregation Attention Network for Remote Sensing Change Detection （2024 TOP）
    中文题目：DGMA2-Net：用于遥感变化检测的差异引导多尺度聚合注意力网络（2024 TOP）
    讲解视频：https://www.bilibili.com/video/BV13ABXYKE86/
        多尺度特征融合单元（Multiscale Feature Fusion,MSFF）：
           思路：利用不同核大小的卷积，MDFM 构建了多尺度融合过程。
           快速梳理：不同的卷积核被用来提取图像的不同部分的特征信息，在通过组合形成多尺度特征图。
           延伸： 1、可以用来增强图像的语义信息，从而提高图像分类、目标检测等任务的性能。
                 2、能够捕捉到不同尺度的特征信息，有效应对图像中存在多尺度目标任务。
'''

class MSFF(nn.Module):
    def __init__(self, inchannel, mid_channel):
        # 调用父类的构造函数
        super(MSFF, self).__init__()
        # 定义第一个卷积序列
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),  # 1x1卷积
            nn.BatchNorm2d(inchannel),  # 批归一化
            nn.ReLU(inplace=True)  # ReLU激活
        )
        # 定义第二个卷积序列，使用3x3卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),  # 1x1卷积降维
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 3, stride=1, padding=1, bias=False),  # 3x3卷积
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),  # 1x1卷积升维
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 定义第三个卷积序列，使用5x5卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 5, stride=1, padding=2, bias=False),  # 5x5卷积
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 定义第四个卷积序列，使用7x7卷积
        self.conv4 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 7, stride=1, padding=3, bias=False),  # 7x7卷积
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 定义混合卷积序列
        self.convmix = nn.Sequential(
            nn.Conv2d(4 * inchannel, inchannel, 1, stride=1, bias=False),  # 1x1卷积降维
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, bias=False),  # 3x3卷积
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 通过不同的卷积序列
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        # 在通道维度上拼接
        x_f = torch.cat([x1, x2, x3, x4], dim=1)
        # 通过混合卷积序列
        out = self.convmix(x_f)

        # 返回输出
        return out

if __name__ == '__main__':
    # 创建一个随机输入
    x = torch.randn((32, 256, 32, 32))
    # 实例化MSFF模型
    model = MSFF(256, 64)
    # 通过模型
    out = model(x)
    # 打印输出形状
    print(out.shape)
    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息