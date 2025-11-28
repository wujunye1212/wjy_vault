import torch
import torch.nn.functional as F
from torch import nn

'''
    论文地址：https://ieeexplore.ieee.org/abstract/document/10504297
    论文题目：DGMA2-Net: A Difference-Guided Multiscale Aggregation Attention Network for Remote Sensing Change Detection （2024 TOP）
    中文题目：DGMA2-Net：用于遥感变化检测的差异引导多尺度聚合注意力网络（2024 TOP）
    讲解视频：https://www.bilibili.com/video/BV1DnzvY9Efd/
        差异增强注意力模块（Difference-Enhanced Attention Module , DEAM）：
           思路：通过建立Q、K和V的自注意力机制结构，建立差异特征与双时相特征之间的关系。
           做法：将差异特征与双时相特征相结合，得到差异增强特征，进一步增强变化区域并细化差异特征，从而提高模型的性能
'''
class DEAM(nn.Module):
    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(DEAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in // 8  # 降低通道数用于计算注意力
        self.activation = activation
        self.ds = ds  # 下采样因子
        self.pool = nn.AvgPool2d(self.ds)  # 平均池化用于下采样
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # 查询卷积
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # 键卷积
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # 值卷积
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习参数用于调整注意力强度
        self.softmax = nn.Softmax(dim=-1)  # softmax用于计算注意力

    def forward(self, input, diff):
        """
            inputs :
                x : 输入特征图 (B X C X W X H)
            returns :
                out : 自注意力值与输入特征相加
                attention: 注意力矩阵 B X N X N (N 是宽度*高度)
        """
        diff = self.pool(diff)  # 对差异图进行下采样
        m_batchsize, C, width, height = diff.size()
        # 计算查询向量
        proj_query = self.query_conv(diff).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        # 计算键向量
        proj_key = self.key_conv(diff).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        # 计算注意力能量
        energy = torch.bmm(proj_query, proj_key)  # 矩阵乘法
        energy = (self.key_channel ** -.5) * energy  # 标准化
        # 计算注意力矩阵
        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        x = self.pool(input)  # 对输入进行下采样
        # 计算值向量
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        # 注意力加权值
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        # 上采样到原始尺寸
        out = F.interpolate(out, [width * self.ds, height * self.ds])
        out = out + input  # 与输入相加

        return out

if __name__ == '__main__':
    x = torch.randn((8, 128, 32, 32))  # 随机生成输入张量
    y = torch.randn((8, 128, 32, 32))  # 随机生成差异张量
    model = DEAM(128)  # 实例化DEAM模型
    out = model(x, y)  # 前向传播
    print(out.shape)  # 输出结果的形状
    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息