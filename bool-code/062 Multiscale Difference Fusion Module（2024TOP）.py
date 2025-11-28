import torch
import torch.nn as nn

'''
    论文地址：https://ieeexplore.ieee.org/abstract/document/10504297
    论文题目：DGMA2-Net: A Difference-Guided Multiscale Aggregation Attention Network for Remote Sensing Change Detection （2024 TOP）
    中文题目：DGMA2-Net：用于遥感变化检测的差异引导多尺度聚合注意力网络（2024 TOP）
    讲解视频：https://www.bilibili.com/video/BV1JczcYzE9j
        多尺度差异融合模块（Multiscale Difference Fusion Module ,MDFM）：
           思路：MDFM 融合从双时相图像获得的特征，并生成了差异特征有价值的上下文信息。
           快速梳理：来自两张配对图像的双时态特征，经过逐像素减法、绝对值运算和3×3卷积运算生成初始差异特征Di。
'''

# 多尺度特征融合模块 (MSFF)
class MSFF(nn.Module):
    def __init__(self, inchannel, mid_channel):
        super(MSFF, self).__init__()
        # 定义不同尺寸的卷积序列
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 混合卷积
        self.convmix = nn.Sequential(
            nn.Conv2d(4 * inchannel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, bias=False),
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
        # 通过混合卷积
        out = self.convmix(x_f)

        return out

# 多尺度差异融合模块 (MDFM)
class MDFM(nn.Module):
    def __init__(self, in_d, out_d):
        super(MDFM, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        # 使用MSFF模块
        self.MPFL = MSFF(inchannel=in_d, mid_channel=64)

        # 差异增强卷积
        self.conv_diff_enh = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )

        # 降维卷积
        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

        # 差异提取卷积
        self.conv_sub = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
        )

        # 混合卷积
        self.convmix = nn.Sequential(
            nn.Conv2d(2 * self.in_d, self.in_d, 3, groups=self.in_d, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        # 差异增强
        x_sub = torch.abs(x1 - x2)
        x_att = torch.sigmoid(self.conv_sub(x_sub))

        x1 = (x1 * x_att) + self.MPFL(self.conv_diff_enh(x1))
        x2 = (x2 * x_att) + self.MPFL(self.conv_diff_enh(x2))

        # 融合
        x_f = torch.stack((x1, x2), dim=2)
        x_f = torch.reshape(x_f, (x1.size(0), -1, x1.size(2), x1.size(3)))
        x_f = self.convmix(x_f)

        # 应用注意力
        x_f = x_f * x_att
        out = self.conv_dr(x_f)

        return out

if __name__ == '__main__':
    # 创建两个随机输入
    x1 = torch.randn((32, 512, 8, 8))
    x2 = torch.randn((32, 512, 8, 8))
    # 实例化MDFM模型
    model = MDFM(512, 64)
    # 通过模型
    out = model(x1, x2)
    # 打印输出形状
    print(out.shape)
    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息