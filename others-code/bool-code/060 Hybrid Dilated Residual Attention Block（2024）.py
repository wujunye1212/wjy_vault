import torch
import torch.nn as nn

'''
    论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0031320324000426
    论文题目：Dual Residual Attention Network for Image Denoising （2024）
    中文题目：用于图像去噪的双残差注意力网络（2024）
    讲解视频：https://www.bilibili.com/video/BV1SjzMYrEGb/
        混合扩张残差注意力（Hybrid Dilated Residual Attention Block，HDRAB）：
           优点：HDRAB由混合扩张残差模块和通道注意模块组成。
                 1、混合扩张残差模块包含混合扩张卷积和RELU，通过多个跳跃连接来捕获局部特征。扩张卷积可以扩大感受野以捕获图像信息。
                 2、通道注意模块（CAM）由全局平均池化（GAP）、Conv、RELU和Sigmoid组成。CAM用于利用卷积特征之间的通道间关系。
'''
class CAB(nn.Module):
    def __init__(self, nc, reduction=8, bias=False):
        # 调用父类的构造函数
        super(CAB, self).__init__()
        # 自适应平均池化，将特征图缩小到1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义通道注意力机制的卷积序列
        self.conv_du = nn.Sequential(
            nn.Conv2d(nc, nc // reduction, kernel_size=1, padding=0, bias=bias),  # 降维
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(nc // reduction, nc, kernel_size=1, padding=0, bias=bias),  # 升维
            nn.Sigmoid()  # Sigmoid激活生成注意力权重
        )

    def forward(self, x):
        # 对输入进行池化
        y = self.avg_pool(x)
        # 通过卷积序列
        y = self.conv_du(y)
        # 将注意力权重应用于输入特征图
        return x * y


class HDRAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        # 调用父类的构造函数
        super(HDRAB, self).__init__()
        # 定义卷积核大小
        kernel_size = 3
        # 通道注意力的降维率
        reduction = 8

        # 定义通道注意力模块
        self.cab = CAB(in_channels, reduction, bias)

        # 定义一系列卷积层和激活层，使用不同的扩张率和填充
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=4, dilation=4, bias=bias)

        self.conv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv_tail = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)

    def forward(self, y):
        # 通过第一组卷积和激活
        y1 = self.conv1(y)
        y1_1 = self.relu1(y1)

        # 通过第二个卷积层并进行跳跃连接
        y2 = self.conv2(y1_1)
        y2_1 = y2 + y

        # 通过第三组卷积和激活
        y3 = self.conv3(y2_1)
        y3_1 = self.relu3(y3)

        # 通过第四个卷积层并进行跳跃连接
        y4 = self.conv4(y3_1)
        y4_1 = y4 + y2_1

        # 通过第五组卷积和激活
        y5 = self.conv3_1(y4_1)
        y5_1 = self.relu3_1(y5)

        # 通过第六个卷积层并进行跳跃连接
        y6 = self.conv2_1(y5_1 + y3)
        y6_1 = y6 + y4_1

        # 通过第七组卷积和激活
        y7 = self.conv1_1(y6_1 + y2_1)
        y7_1 = self.relu1_1(y7)

        # 通过尾部卷积层并进行跳跃连接
        y8 = self.conv_tail(y7_1 + y1)
        y8_1 = y8 + y6_1

        # 通过通道注意力模块
        y9 = self.cab(y8_1)
        y9_1 = y + y9

        # 返回最终结果
        return y9_1


if __name__ == '__main__':
    # 假输入
    input_tensor = torch.randn(1, 64, 256, 256)

    # 实例化HDRAB
    hdrab = HDRAB(in_channels=64, out_channels=64, bias=True)
    # 通过HDRAB
    hdrab_output = hdrab(input_tensor)
    print("HDRAB 输入维度:", input_tensor.shape)
    print("HDRAB 输出维度:", hdrab_output.shape)
    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息