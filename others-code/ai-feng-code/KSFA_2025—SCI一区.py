import torch
import torch.nn as nn
from einops import rearrange
#来自2025 SCI一区
'''
Ai缝合怪提示：KSFA模块内容讲解，可以看b站对应视频

'''
class KSFA(nn.Module):
    def __init__(self, dim, r=16, L=32): # 初始化函数，接收输入特征图的维度dim，缩放因子r，默认值为16，最小瓶颈通道数L，默认值为32
        super().__init__() # 调用父类的初始化方法
        d = max(dim // r, L) # 计算中间层的通道数d，取dim除以r和L的最大值，确保不会压缩得太多
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim) # 使用深度卷积（groups=dim）进行3x3卷积，提取局部空间信息
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2) # 使用膨胀率为2的5x5卷积扩大感受野
        self.conv1 = nn.Conv2d(dim, dim // 2, 1) # 1x1卷积将通道数减半
        self.conv2 = nn.Conv2d(dim, dim // 2, 1) # 同上，用于第二个分支
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3) # 对平均池化和最大池化的结果进行7x7卷积处理，生成空间注意力权重
        self.conv = nn.Conv2d(dim // 2, dim, 1) # 最终输出前恢复通道数

        self.global_pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化，用于提取通道注意力的全局特征
        self.global_maxpool = nn.AdaptiveMaxPool2d(1) # 全局最大池化，用于提取通道注意力的全局特征
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False), # 第一层FC：将通道数从dim压缩到d
            nn.BatchNorm2d(d), # 批归一化
            nn.ReLU(inplace=True) # ReLU激活函数
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False) # 第二层FC：恢复通道数到原始dim，用于生成注意力权重
        self.softmax = nn.Softmax(dim=1) # softmax用于在两个分支之间分配注意力权重

    def forward(self, x): # 前向传播函数
        batch_size = x.size(0) # 获取batch size
        dim = x.size(1) # 获取通道数
        attn1 = self.conv0(x)  # conv_3*3，使用3x3卷积提取局部特征
        attn2 = self.conv_spatial(attn1)  # conv_3*3 -> conv_5*5，在attn1基础上再经过膨胀卷积，获得更大感受野的特征

        attn1 = self.conv1(attn1) # b, dim/2, h, w，对第一个分支的输出降维为dim//2通道
        attn2 = self.conv2(attn2) # b, dim/2, h, w，对第二个分支的输出降维为dim//2通道

        attn = torch.cat([attn1, attn2], dim=1)  # b,c,h,w，按通道拼接两个分支的特征图
        avg_attn = torch.mean(attn, dim=1, keepdim=True) # b,1,h,w，求平均
        max_attn, _ = torch.max(attn, dim=1, keepdim=True) # b,1,h,w，计算最大值
        agg = torch.cat([avg_attn, max_attn], dim=1) # b,2,h,w，拼接平均和最大的结果

        ch_attn1 = self.global_pool(attn) # b,dim,1, 1，全局平均池化  ，作者希望大家多去跑跑实验看看哪一种池化效果好
        z = self.fc1(ch_attn1) # 通过第一层FC网络生成通道注意力
        a_b = self.fc2(z) # 恢复通道数到原始dim
        a_b = a_b.reshape(batch_size, 2, dim // 2, -1) # 调整形状以便后续乘法操作
        a_b = self.softmax(a_b)

        a1,a2 =  a_b.chunk(2, dim=1) # 将通道注意力拆分为两个分支
        a1 = a1.reshape(batch_size,dim // 2,1,1) # 调整a1的形状
        a2 = a2.reshape(batch_size, dim // 2, 1, 1) # 调整a2的形状

        w1 = a1 * agg[:, 0, :, :].unsqueeze(1) # 将通道注意力和空间注意力结合，得到加权系数w1
        w2 = a2 * agg[:, 0, :, :].unsqueeze(1) # 将通道注意力和空间注意力结合，得到加权系数w2

        attn = attn1 * w1 + attn2 * w2 # 对两个分支的特征图进行加权融合
        attn = self.conv(attn).sigmoid() # 恢复通道数并用sigmoid激活，得到注意力权重图
        return x * attn # 将注意力权重作用到原始输入特征图上，完成特征增强

if __name__ == '__main__':
    # 创建随机输入张量，形状为 (batch_size, dim, height, width)
    input = torch.randn(2, 32, 64,64)
    model = KSFA(dim=32)
    # 进行前向传播，得到输出
    output = model(input)
    # 打印输入和输出的形状
    print('AI缝合怪-KSFA_Input size:', input.size())  # 打印输入张量的形状
    print('AI缝合怪-KSFA_Output size:', output.size())  # 打印输出张量的形状
