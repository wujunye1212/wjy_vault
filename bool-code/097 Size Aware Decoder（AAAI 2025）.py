import torch.nn as nn
import torch
"""
    论文地址：https://arxiv.org/pdf/2412.08345
    论文题目：ConDSeg: A General Medical Image Segmentation Framework via Contrast-Driven Feature Enhancement (AAAI 2025)
    中文题目：A2RNet：具有对抗攻击鲁棒性的红外和可见光图像融合网络 (AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV1nLrdYvEto/
        尺寸感知解码器（Size-Aware Decoder, SA-Decoder）
             理论研究：根据不同层次特征图包含不同粒度信息的特点，建立三个解码器分别用于预测小、中、大尺寸实体，各解码器接收相邻CDFA输出的特征图，最后融合三个解码器的输出得到最终掩码。
"""
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        # 初始化CBR模块，设置卷积层参数和激活层开关
        self.act = act

        # 定义卷积层和批归一化层
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        # 定义ReLU激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 前向传播，先经过卷积和批归一化
        x = self.conv(x)
        # 如果激活开关打开，则通过ReLU激活函数
        if self.act == True:
            x = self.relu(x)
        return x

class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        # 初始化通道注意力模块，定义自适应池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 定义全连接层和激活函数
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        # 定义Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        # 通过平均池化和最大池化计算注意力
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 合并注意力结果
        out = avg_out + max_out
        # 返回加权输出
        return x0 * self.sigmoid(out)

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        # 确保卷积核大小为3或7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 定义卷积层和Sigmoid激活函数
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x  # 输入张量 [B,C,H,W]
        # 计算平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接池化结果
        x = torch.cat([avg_out, max_out], dim=1)
        # 通过卷积和激活函数
        x = self.conv1(x)
        # 返回加权输出
        return x0 * self.sigmoid(x)

class SizeAwareDecoder(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        # 初始化SizeAwareDecoder，定义上采样和CBR模块
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

        # 定义上采样层
        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.c1 = CBR(in_c + out_c, out_c, kernel_size=1, padding=0)
        self.c2 = CBR(out_c, out_c, act=False)
        self.c3 = CBR(out_c, out_c, act=False)
        self.c4 = CBR(out_c, out_c, kernel_size=1, padding=0, act=False)

        # 定义通道和空间注意力模块
        self.ca = channel_attention(out_c)
        self.sa = spatial_attention()

    def forward(self, x, skip):
        # 上采样并拼接跳跃连接
        # 如果是两个相同大小，那么把这行代码注释掉就行
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)

        # 通过CBR模块
        x = self.c1(x)

        # 残差连接和激活
        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        s3 = x
        x = self.c4(x)
        x = self.relu(x + s3 + s2 + s1)

        # 通过通道和空间注意力模块
        x = self.ca(x)
        x = self.sa(x)
        return x


if __name__ == "__main__":
    # 下采样 融合
    # 创建输入张量
    x = torch.randn(1, 32, 64 // 2, 64 // 2)  # 下采样的输入
    skip = torch.randn(1, 64, 64, 64)  # 跳跃连接

    size_aware_decoder = SizeAwareDecoder(in_c=64, out_c=32, scale=2)

    output = size_aware_decoder(x, skip)
    print("输出形状:", output.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")