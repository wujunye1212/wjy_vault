import torch
from torch import nn

'''
    论文地址：https://dl.acm.org/doi/abs/10.1145/3664647.3680650
    论文题目：Multi-Scale and Detail-Enhanced Segment Anything Model for Salient Object Detection（A会议 2024）
    中文题目：多尺度和细节增强的任意分割模型（A会议 2024）
    讲解视频：https://www.bilibili.com/video/BV1PTDoYAEhx/
        多尺度边缘增强模块 (Multi-scale Edge Enhancement Module)
            作用：旨在通过深度学习技术增强图像中的边缘特征。
            1、EdgeEnhancer：利用不同的卷积和池化操作，提取出图像中的边缘特征，专注于不同的尺度，从而捕捉到多层次的边缘信息。
                            思想来源 图像差分 参考网址 https://blog.csdn.net/qq_36332660/article/details/134684343
            2、MEEM        ：经过多层处理后，将提取到的边缘特征与原始输入进行融合，有助于保留原始图像的信息，同时突出边缘特征，增强图像的细节和清晰度。
'''
class EdgeEnhancer(nn.Module):  # 边缘增强模块
    def __init__(self, in_dim, norm, act):  # 初始化函数，接收输入维度、归一化层和激活函数
        super().__init__()  # 调用父类构造函数
        self.out_conv = nn.Sequential(  # 定义输出卷积层
            nn.Conv2d(in_dim, in_dim, 1, bias=False),  # 1x1卷积，不使用偏置
            norm(in_dim),  # 归一化层
            nn.Sigmoid()  # Sigmoid激活函数  : 将输出限制在0 到1 的范围内，有助于将增强的边缘特征与原始图像合并时保持一定的平衡，防止过强的增强导致信息丢失。
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)  # 定义平均池化层

    def forward(self, x):  # 前向传播函数
        """
            首先经过平均池化操作，这会平滑图像并降低细节。
            然后，通过计算输入图像与池化结果之间的差异（edge = x - edge），可以提取出图像的边缘信息。
            边缘通常是图像中像素值变化较大的地方，因此这种差异计算有助于强调边缘特征。
        """
        edge = self.pool(x)  # 对输入进行池化操作
        edge = x - edge  # 计算边缘信息，提取出图像的边缘信息
        edge = self.out_conv(edge)  # 通过输出卷积层处理边缘信息
        # 【通过残差 强化细节】
        return x + edge

class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim  # 输入维度
        self.hidden_dim = hidden_dim  # 隐藏层维度

        """
            self.width 可以简单理解为 EdgeEnhancer重复的次数
        """
        self.width = width

        self.in_conv = nn.Sequential(  # 定义输入卷积层
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),  # 1x1卷积，不使用偏置
            norm(hidden_dim),  # 归一化层
            nn.Sigmoid()  # Sigmoid激活函数
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)  # 定义平均池化层

        self.mid_conv = nn.ModuleList()  # 中间卷积层列表
        self.edge_enhance = nn.ModuleList()  # 边缘增强模块列表

        for i in range(width - 1):  # 遍历宽度
            self.mid_conv.append(nn.Sequential(  # 添加中间卷积层
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),  # 1x1卷积，不使用偏置
                norm(hidden_dim),  # 归一化层
                nn.Sigmoid()  # Sigmoid激活函数
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))  # 添加边缘增强模块

        self.out_conv = nn.Sequential(  # 定义输出卷积层
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),  # 1x1卷积，不使用偏置
            norm(in_dim),  # 归一化层
            act()  # 激活函数
        )

    def forward(self, x):  # 前向传播函数
        mid = self.in_conv(x)  # 通过输入卷积层处理输入

        out = mid  # 初始化输出为中间结果

        for i in range(self.width - 1):  # 遍历宽度
            mid = self.pool(mid)        # 对中间结果进行池化

            mid = self.mid_conv[i](mid)  # 通过中间卷积层处理
            edge = self.edge_enhance[i](mid)
            out = torch.cat([out,edge], dim=1)  # 多个 EdgeEnhancer 实例被用于处理经过多次卷积和池化的特征图。

            # 这种多层次的处理方式使得边缘信息可以在不同的尺度上被提取和增强，进一步提高了模型对边缘特征的敏感性。

        out = self.out_conv(out)  # 通过输出卷积层处理最终结果

        return out  # 返回最终输出

if __name__ == '__main__':

    MEEM = MEEM(in_dim=64,hidden_dim=32)
    input = torch.randn(1, 64, 128, 128)

    output = MEEM(input)
    print('Input size:', input.size())
    print('Output size:', output.size())

    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息