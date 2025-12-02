import torch
from torch import nn
"""
    论文地址：https://arxiv.org/pdf/2107.05274
    论文题目：TransAttUnet: Multi-level Attention-guided U-Net with Transformer for Medical Image Segmentation（TETCI 2024）
    中文题目：TransAttUnet：用于医学图像分割的具有变换器的多级注意力引导U-Net （TETCI 2024）
    讲解视频：https://www.bilibili.com/video/BV1yQ91YVETU/
        全局空间注意力（Global Spatial Attention，GSA）：
            实际意义：①局部特征局限性：在图像分析任务中，如医学图像分割、自然图像语义分割等，仅依靠局部特征难以准确识别和分割目标。
                    ②同类特征紧凑性：需要模型对同一类别特征有更好的聚类和区分能力。③长距离依赖关系：目标之间存在长距离依赖关系，传统方法难以捕捉。
            实现方式：首先，通过卷积生成特征图，接着计算位置注意力图，最终融合全局信息生成最终特征。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""
# 空间注意力模块 (Global Spatial Attention, GSA)
class Global_Spatial_Attention(nn.Module):
    def __init__(self, in_dim):
        super(Global_Spatial_Attention, self).__init__()
        self.chanel_in = in_dim  # 输入通道数

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # 查询卷积
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # 键卷积
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # 值卷积

        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放参数
        self.softmax = nn.Softmax(dim=-1)  # 对注意力权重进行归一化

    def forward(self, x):
        m_batchsize, C, height, width = x.size()  # 获取输入张量的批量大小、通道数、高度和宽度

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # 计算查询向量并调整形状
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # 计算键向量并调整形状

        energy = torch.bmm(proj_query, proj_key)  # 计算查询和键的点积，得到注意力能量
        attention = self.softmax(energy)  # 对注意力能量进行归一化，得到注意力权重

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # 计算值向量并调整形状
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # 根据注意力权重加权值向量
        out = out.view(m_batchsize, C, height, width)  # 恢复输出张量的形状

        out = self.gamma * out + x  # 将注意力输出与输入残差连接
        return out  # 返回最终的输出

if __name__ == '__main__':
    input = torch.rand(1, 64, 128, 128)  # 创建一个随机输入张量
    SAA = Global_Spatial_Attention(in_dim=64)  # 初始化空间注意力模块
    output = SAA(input)  # 计算空间注意力模块的输出
    print("input.shape:", input.shape)  # 打印输入张量的形状
    print("output.shape:", output.shape)  # 打印输出张量的形状
    print("公众号、B站、CSDN同号")  # 输出推广信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 输出提示信息
