import torch
import torch.nn as nn
"""
    论文地址：https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0303670
    论文题目：DAU-Net: Dual attention-aided U-Net for segmenting tumor in breast ultrasound images
    中文题目：DAU-Net：用于乳腺超声图像中肿瘤分割的双重注意力辅助 U-Net
    讲解视频：https://www.bilibili.com/video/BV1JYqqYiEum/
        1、CBAM 通过通道注意力（CAM）和空间注意力（SAM）捕获上下文感知特征和空间关系。
        2、PAM 通过卷积层和注意力机制丰富局部特征并捕获空间关系，两者结合增强了模型对局部特征的表示能力。
"""

class ChannelAttentionModule(nn.Module):  # 定义通道注意力模块
    def __init__(self, in_channels, ratio=8):  # 初始化函数，设置输入通道数和缩放比例
        super(ChannelAttentionModule, self).__init__()  # 调用父类的初始化方法
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 定义自适应平均池化层
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 定义自适应最大池化层

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False)  # 定义第一个全连接卷积层
        self.relu1 = nn.ReLU()  # 定义 ReLU 激活函数
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False)  # 定义第二个全连接卷积层

    def forward(self, x):  # 前向传播函数
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 通过平均池化计算通道注意力
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 通过最大池化计算通道注意力
        out = avg_out + max_out  # 将平均池化和最大池化的结果相加
        return x * torch.sigmoid(out)  # 用 sigmoid 激活函数调整输入并返回

class SpatialAttentionModule(nn.Module):  # 定义空间注意力模块
    def __init__(self):  # 初始化函数
        super(SpatialAttentionModule, self).__init__()  # 调用父类的初始化方法
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)  # 定义卷积层

    def forward(self, x):  # 前向传播函数
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 计算输入的平均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 计算输入的最大值
        out = torch.cat([avg_out, max_out], dim=1)  # 将平均值和最大值拼接在一起
        out = self.conv1(out)  # 通过卷积层
        return x * torch.sigmoid(out)  # 用 sigmoid 激活函数调整输入并返回

class PAM(nn.Module):  # 定义位置注意力模块
    def __init__(self, in_channels):  # 初始化函数
        super(PAM, self).__init__()  # 调用父类的初始化方法
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)  # 定义查询卷积
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)  # 定义键卷积
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 定义值卷积
        self.gamma = nn.Parameter(torch.zeros(1))  # 定义可训练参数 gamma
        self.softmax = nn.Softmax(dim=-1)  # 定义 softmax 函数

    def forward(self, x):  # 前向传播函数
        batch_size, C, height, width = x.size()  # 获取输入张量的尺寸

        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # 计算查询向量并调整形状 Q
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)  # 计算键向量 K
        energy = torch.bmm(proj_query, proj_key)  # 计算能量矩阵
        attention = self.softmax(energy)  # 对能量矩阵应用 softmax

        proj_value = self.value_conv(x).view(batch_size, -1, height * width)  # 计算值向量 V
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # 计算加权值向量
        out = out.view(batch_size, C, height, width)  # 调整输出形状

        out = self.gamma * out + x  # 将加权输出与输入相加

        return out  # 返回最终输出

class PCBAM(nn.Module):  # 定义 PCBAM 模块
    def __init__(self, in_channels, ratio=8):  # 初始化函数
        super(PCBAM, self).__init__()  # 调用父类的初始化方法
        self.channel_attention = ChannelAttentionModule(in_channels, ratio)  # 定义通道注意力模块
        self.spatial_attention = SpatialAttentionModule()  # 定义空间注意力模块
        self.position_attention = PAM(in_channels)  # 定义位置注意力模块

    def forward(self, x):  # 前向传播函数
        # CBAM
        x_c = self.channel_attention(x)  # 通过通道注意力模块
        x_s = self.spatial_attention(x_c)  # 通过空间注意力模块

        x_p = self.position_attention(x)  # 通过位置注意力模块

        out = x_s + x_p  # 将空间和位置注意力的结果相加
        return out  # 返回最终输出

if __name__ == '__main__':
    # 假设输入张量形状为 (batch_size, in_channels, height, width)
    batch_size = 4
    in_channels = 64
    height = 32
    width = 32

    # 随机生成一个输入张量
    x = torch.randn(batch_size, in_channels, height, width)
    pcbam = PCBAM(in_channels=in_channels)

    print("Input shape:", x.shape)
    out_pcbam = pcbam(x)
    print("Output shape:", out_pcbam.shape)







