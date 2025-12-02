import torch
import torch.nn as nn

'''
    论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0031320324000426
    论文题目：Dual Residual Attention Network for Image Denoising （2024）
    中文题目：用于图像去噪的双残差注意力网络（2024）
    讲解视频：https://www.bilibili.com/video/BV19pBYYSEKy/
        残差注意力（Residual Attention Block ,RAB）：
           思路：RAB由残差模块和空间注意模块（SAM）组成。
                1、残差模块包括标准卷积和RELU。通过卷积层之间的多个跳跃连接可以提取丰富的局部特征。
                2、卷积层只能提取局部信息而不能利用非局部信息，因此采用注意力机制来捕获全局上下文信息。
           优点：与现有的残差注意模块相比，拥有更多的跳跃连接，可以提取和融合不同卷积层之间的特征，进一步提高去噪性能。
'''
class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=False):
        # 调用父类的构造函数
        super(Basic, self).__init__()
        # 输出通道数
        self.out_channels = out_planes
        # 卷积层的组数，默认为1
        groups = 1
        # 定义二维卷积层
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 通过卷积层
        x = self.conv(x)
        # 通过ReLU激活函数
        x = self.relu(x)
        # 返回结果
        return x

class ChannelPool(nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super(ChannelPool, self).__init__()

    def forward(self, x):
        # 在通道维度上取最大值和平均值，并在新的维度上拼接
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SAB(nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super(SAB, self).__init__()
        # 定义卷积核大小
        kernel_size = 5
        # 定义通道池化操作
        self.compress = ChannelPool()
        # 定义空间注意力机制中的卷积操作
        self.spatial = Basic(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        # 压缩通道信息
        x_compress = self.compress(x)
        # 通过空间卷积
        x_out = self.spatial(x_compress)
        # 通过Sigmoid函数生成注意力图
        scale = torch.sigmoid(x_out)
        # 返回加权后的特征图
        return x * scale


class RAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        # 调用父类的构造函数
        super(RAB, self).__init__()
        # 定义卷积核大小
        kernel_size = 3
        # 定义步长
        stride = 1
        # 定义填充
        padding = 1
        # 初始化卷积层列表
        layers = []
        # 添加第一个卷积层和ReLU激活
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        # 添加第二个卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        # 将卷积层组成一个序列
        self.res = nn.Sequential(*layers)
        # 定义空间注意力模块
        self.sab = SAB()

    def forward(self, x):
        # 第一次残差连接
        x1 = x + self.res(x)
        # 第二次残差连接
        x2 = x1 + self.res(x1)
        # 第三次残差连接
        x3 = x2 + self.res(x2)

        # 叠加x1和x3
        x3_1 = x1 + x3
        # 第四次残差连接
        x4 = x3_1 + self.res(x3_1)
        # 叠加x和x4
        x4_1 = x + x4

        # 通过空间注意力模块
        x5 = self.sab(x4_1)
        # 叠加x和x5
        x5_1 = x + x5

        # 返回最终结果
        return x5_1

if __name__ == '__main__':
    # 假输入
    input_tensor = torch.randn(1, 64, 256, 256)

    # 实例化RAB
    rab = RAB(in_channels=64, out_channels=64, bias=True)
    # 通过RAB
    rab_output = rab(input_tensor)
    print("RAB 输入维度:", input_tensor.shape)
    print("RAB 输出维度:", rab_output.shape)
    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息