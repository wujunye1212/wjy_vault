import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2412.08345
    论文题目：ConDSeg: A General Medical Image Segmentation Framework via Contrast-Driven Feature Enhancement (AAAI 2025)
    中文题目：A2RNet：具有对抗攻击鲁棒性的红外和可见光图像融合网络 (AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV1fsrqY1EvB/
        空洞卷积层增强模块 
                用途：增强特征表示
"""
# 定义卷积、批归一化和激活函数的模块
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act  # 是否使用激活函数

        # 定义卷积和批归一化的顺序
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)  # 定义ReLU激活函数

    def forward(self, x):
        x = self.conv(x)  # 执行卷积和批归一化
        if self.act == True:  # 如果需要激活函数
            x = self.relu(x)  # 执行ReLU激活
        return x  # 返回结果

# 定义通道注意力模块
class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 定义全连接层，使用1x1卷积实现
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()  # 定义ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()  # 定义Sigmoid激活函数

    def forward(self, x):
        x0 = x  # 保存输入
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 平均池化分支
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 最大池化分支
        out = avg_out + max_out  # 合并两个分支
        return x0 * self.sigmoid(out)  # 返回加权后的输入

# 定义空间注意力模块
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 确保核大小为3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核大小设置填充

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 定义卷积层
        self.sigmoid = nn.Sigmoid()  # 定义Sigmoid激活函数

    def forward(self, x):
        x0 = x  # [B,C,H,W] 保存输入
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 合并两个池化结果
        x = self.conv1(x)  # 卷积操作
        return x0 * self.sigmoid(x)  # 返回加权后的输入

# 定义膨胀卷积模块
class dilated_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)  # 定义ReLU激活函数

        # 定义多个膨胀卷积和通道注意力模块
        self.c1 = nn.Sequential(CBR(in_c, out_c, kernel_size=1, padding=0), channel_attention(out_c))
        self.c2 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=6, dilation=6), channel_attention(out_c))
        self.c3 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=12, dilation=12), channel_attention(out_c))
        self.c4 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=18, dilation=18), channel_attention(out_c))
        self.c5 = CBR(out_c * 4, out_c, kernel_size=3, padding=1, act=False)  # 合并后卷积
        self.c6 = CBR(in_c, out_c, kernel_size=1, padding=0, act=False)  # 直接卷积

        self.sa = spatial_attention()  # 定义空间注意力模块

    def forward(self, x):
        x1 = self.c1(x)  # 第一个膨胀卷积
        x2 = self.c2(x)  # 第二个膨胀卷积
        x3 = self.c3(x)  # 第三个膨胀卷积
        x4 = self.c4(x)  # 第四个膨胀卷积
        xc = torch.cat([x1, x2, x3, x4], axis=1)  # 合并所有膨胀卷积的输出
        xc = self.c5(xc)  # 卷积处理合并后的输出

        xs = self.c6(x)  # 直接卷积处理输入

        x = self.relu(xc + xs)  # 合并并激活
        x = self.sa(x)  # 空间注意力处理
        return x  # 返回结果

if __name__ == '__main__':
    input = torch.rand(1, 64, 32, 32)  # 创建一个随机输入张量
    drm = dilated_conv(64,64)  # 实例化膨胀卷积模块
    output = drm(input)  # 获取输出
    print("DRM_input.shape:", input.shape)  # 打印输入形状
    print("DRM_output.shape:", output.shape)  # 打印输出形状
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")