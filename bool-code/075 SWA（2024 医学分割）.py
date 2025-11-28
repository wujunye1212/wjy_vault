import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    论文地址：https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0303670
    论文题目：DAU-Net: Dual attention-aided U-Net for segmenting tumor in breast ultrasound images
    中文题目：DAU-Net：用于乳腺超声图像中肿瘤分割的双重注意力辅助 U-Net
    讲解视频：https://www.bilibili.com/video/BV1JYqqYiEum/
        基于窗口的注意力机制
            通过计算位置感知的查询、键和值矩阵生成注意力图，然后与原始特征图相加，使模型能够关注图像不同部分的相关信息，
                            有效捕获全局依赖关系，提高分割结果的空间连贯性。
"""
class SWA(nn.Module):  # 定义一个名为 SWA 的类，继承自 nn.Module
    def __init__(self, in_channels, n_heads=8, window_size=7):  # 初始化函数，设置输入通道数、头数和窗口大小
        super(SWA, self).__init__()  # 调用父类的初始化方法
        self.in_channels = in_channels  # 保存输入通道数
        self.n_heads = n_heads  # 保存多头注意力机制的头数
        self.window_size = window_size  # 保存窗口大小

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 定义查询卷积
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 定义键卷积
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 定义值卷积
        self.gamma = nn.Parameter(torch.zeros(1))  # 定义一个可训练参数 gamma，初始为 0
        self.softmax = nn.Softmax(dim=-1)  # 定义 softmax 函数，用于计算注意力分数

    def forward(self, x):  # 前向传播函数
        batch_size, C, height, width = x.size()  # 获取输入张量的尺寸

        padded_x = F.pad(x, [self.window_size // 2, self.window_size // 2,
                             self.window_size // 2, self.window_size // 2], mode='reflect')  # 对输入进行反射填充

        proj_query = self.query_conv(x).view(batch_size, self.n_heads,C // self.n_heads, height * width)  # 计算查询向量并调整形状

        proj_key = self.key_conv(padded_x).unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)  # 计算键向量并展开
        proj_key = proj_key.permute(0, 1, 4, 5, 2, 3).contiguous().view(batch_size,self.n_heads, C // self.n_heads, -1)  # 调整键向量形状

        proj_value = self.value_conv(padded_x).unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)  # 计算值向量并展开
        proj_value = proj_value.permute(0, 1, 4, 5, 2, 3).contiguous().view(batch_size,
                                                                            self.n_heads, C // self.n_heads, -1)  # 调整值向量形状

        energy = torch.matmul(proj_query.permute(0, 1, 3, 2), proj_key)  # 计算查询和键之间的能量
        attention = self.softmax(energy)  # 对能量应用 softmax，得到注意力得分

        out_window = torch.matmul(attention, proj_value.permute(0, 1, 3, 2))  # 计算加权值向量
        out_window = out_window.permute(0, 1, 3, 2).contiguous().view(batch_size, C, height, width)  # 调整输出形状

        out = self.gamma * out_window + x  # 将加权输出与输入相加
        return out  # 返回最终输出


if __name__ == '__main__':
    batch_size = 4
    in_channels = 64
    height = 32
    width = 32

    x = torch.randn(batch_size, in_channels, height, width)
    swa = SWA(in_channels=in_channels)

    print("Input shape:", x.shape)
    out_swa = swa(x)
    print("Output shape:", out_swa.shape)
