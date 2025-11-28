import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/abs/2304.08069
    论文题目：D-Net: Dynamic Large Kernel with Dynamic Feature Fusion for Volumetric Medical Image Segmentation（arxiv 2024）
    中文题目：D-Net：用于体积医学图像分割的动态大核与动态特征融合（arxiv 2024）
    讲解视频：https://www.bilibili.com/video/BV1ukzRY3Ecn/
    3D动态大核卷积（Dynamic Large Kernel ,DLK）：
         作用：具有更大的感受野和更长的感受野，能够更好地捕捉图像中的上下文信息。
         结构组成：多个大核可变卷积核来捕获多尺度上下文信息，引入空间级动态选择机制以根据全局上下文信息自适应地选择最重要的局部特征。
"""
class DLK(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 初始化第一个深度可分离卷积层
        self.att_conv1 = nn.Conv3d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        # 初始化第二个深度可分离卷积层，具有扩张卷积
        self.att_conv2 = nn.Conv3d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)

        # 初始化空间注意力模块
        self.spatial_se = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通过第一个卷积层
        att1 = self.att_conv1(x)
        # 通过第二个卷积层
        att2 = self.att_conv2(att1)

        # 将两个卷积的结果在通道维度上拼接
        att = torch.cat([att1, att2], dim=1)
        # 计算平均注意力
        avg_att = torch.mean(att, dim=1, keepdim=True)
        # 计算最大注意力
        max_att, _ = torch.max(att, dim=1, keepdim=True)
        # 将平均和最大注意力拼接
        att = torch.cat([avg_att, max_att], dim=1)
        # 通过空间注意力模块
        att = self.spatial_se(att)

        # 计算输出，融合注意力
        output = att1 * att[:, 0, :, :, :].unsqueeze(1) + att2 * att[:, 1, :, :, :].unsqueeze(1)
        # 将输入加回输出
        output = output + x

        return output

if __name__ == '__main__':
    # 设置通道维度
    dim = 64
    # 创建 DLK 实例
    dlk = DLK(dim)

    # 设置输入张量的形状
    B, D, H, W = 2, 8, 32, 32  # 示例 batch size 和空间维度
    x = torch.randn(B, dim, D, H, W)

    # 调用 forward 方法
    output = dlk(x)

    # 打印输出的形状
    print(output.shape)

    # 打印社交媒体账号信息
    print("抖音、B站、小红书、CSDN同号")
    # 打印提醒信息
    print("布尔大学士 提醒您：代码无误~~~~")
