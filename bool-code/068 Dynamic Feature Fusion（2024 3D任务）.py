import torch
import torch.nn as nn
"""
    论文地址：https://arxiv.org/abs/2304.08069
    论文题目：D-Net: Dynamic Large Kernel with Dynamic Feature Fusion for Volumetric Medical Image Segmentation（arxiv 2024）
    中文题目：D-Net：用于体积医学图像分割的动态大核与动态特征融合（arxiv 2024）
    讲解视频：https://www.bilibili.com/video/BV1X3idYzEAB/
        动态特征融合（Dynamic Feature Fusion,DFF）：
             作用：根据全局信息动态选择重要的特征并将其融合。
             结构组成：通过通道级和空间级动态选择机制来实现自适应融合多尺度特征。
"""
class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 自适应平均池化，将输出尺寸调整为1x1x1
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # 注意力卷积层，使用Sigmoid激活
        self.conv_atten = nn.Sequential(
            nn.Conv3d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 通道缩减卷积层
        self.conv_redu = nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False)
        # 单通道卷积层，用于计算注意力
        self.conv1 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        # Sigmoid激活函数
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):

        # 【图中下半部分】
        # 将输入和跳跃连接的特征拼接在一起
        output = torch.cat([x, skip], dim=1)
        # 计算注意力权重
        att = self.conv_atten(self.avg_pool(output))
        # 应用注意力权重
        output = output * att
        # 通道缩减
        output = self.conv_redu(output)

        # 【图中上半部分】
        # 计算新的注意力
        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)

        # 应用新的注意力权重
        output = output * att

        return output

if __name__ == '__main__':
    # 创建两个输入张量
    input1 = torch.randn(1, 32, 16, 64, 64) # x: (B, C, D, H, W) 3D图像维度
    input2 = torch.randn(1, 32, 16, 64, 64) # x: (B, C, D, H, W) 3D图像维度
    # 实例化DFF模型
    model = DFF(32)
    # 计算输出
    output = model(input1, input2)
    # 打印输入和输出的尺寸
    print("DFF_input size:", input1.size())
    print("DFF_Output size:", output.size())

    # 打印社交媒体账号信息
    print("抖音、B站、小红书、CSDN同号")
    # 打印提醒信息
    print("布尔大学士 提醒您：代码无误~~~~")

