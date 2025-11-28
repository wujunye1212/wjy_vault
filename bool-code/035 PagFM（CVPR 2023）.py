import torch
import torch.nn as nn
import torch.nn.functional as F
'''
    论文地址：https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html
    论文题目：PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers（CVPR 2023）
    中文题目：受 PID 控制器启发的实时语义分割网络
    讲解视频：https://www.bilibili.com/video/BV1twykYmEnn/
         选择性地学习高级语义（Pag: Learning High-level Semantics）：
            解决：高分辨率细节与低频上下文直接融合导致细节特征容易被周围环境信息淹没。
'''
class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=True, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()

        # 是否包含通道注意力机制
        self.with_channel = with_channel
        # 是否在输入后应用ReLU激活函数
        self.after_relu = after_relu

        # 定义用于处理输入x的卷积序列f_x
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        # 定义用于处理输入y的卷积序列f_y
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )

        # 如果配置中包含通道注意力，则定义上采样卷积序列
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels,kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )

        # 如果配置要求在输入后应用ReLU激活函数，则定义ReLU层
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()

        # 如果配置要求在输入后应用ReLU激活函数，则对输入x和y应用ReLU
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        # 对输入y应用卷积序列f_y得到y_q
        y_q = self.f_y(y)
        # 使用双线性插值法调整y_q的尺寸以匹配输入x的尺寸 [由第一个输入特征的大小决定]
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],mode='bilinear', align_corners=False)

        # 对输入x应用卷积序列f_x得到x_k
        x_k = self.f_x(x)

        if self.with_channel:
            # 使用sigmoid激活函数处理经过上采样后的x_k和y_q的乘积
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            # 计算x_k和y_q的点积然后求和，并使用sigmoid激活函数
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        # 使用双线性插值法调整y的尺寸以匹配输入x的尺寸 [由第一个输入特征的大小决定]
        y = F.interpolate(y, size=[input_size[2], input_size[3]],mode='bilinear', align_corners=False)
        # 根据相似度图sim_map更新输入x
        x = (1 - sim_map) * x + sim_map * y

        return x

# 主程序入口
if __name__ == '__main__':
    block = PagFM(64,32)

    # input1 = torch.rand(3, 64, 128, 128)
    input1 = torch.rand(3, 64, 32, 32)

    input2 = torch.rand(3, 64, 64, 64)

    output = block(input1, input2)

    # 打印输入和输出张量的形状。
    print(f"Input1 shape: {input1.shape}")
    print(f"Input2 shape: {input2.shape}")

    print(f"Output shape: {output.shape}")

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")