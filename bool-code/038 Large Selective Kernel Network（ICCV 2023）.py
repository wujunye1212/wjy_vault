import torch
import torch.nn as nn

'''
    论文地址：https://arxiv.org/pdf/2303.09030.pdf
    论文题目：LSKNet: Large Selective Kernel Network for Remote Sensing Object Detection（ICCV 2023）
    中文题目：用于遥感目标检测的选择性大核网络
    讲解视频：https://www.bilibili.com/video/BV1baSqYnEdP/
         选择性大核网络（Large Selective Kernel Network,LSKNet)：
            首先，通过大核卷积明确产生具有各种大感受野特征。其次，为了增强网络对检测目标最相关空间上下文区域的关注能力，使用空间选择机制来从不同尺度的大卷积核中进行空间选择特征图。
'''


class Large_Selective_Kernel_Network(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # 定义一个大小为5x5的深度可分离卷积层，输入输出通道数均为dim，padding为2
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)  # 定义一个具有3倍膨胀率的空间卷积层，大小为7x7，步长为1，padding为9

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)  # 定义一个大小为1x1的卷积层，将通道数减半
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)  # 同上，定义另一个大小为1x1的卷积层，将通道数减半

        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)  # 定义一个大小为7x7的卷积层用于压缩特征图，输入通道数为2，输出通道数也为2
        self.conv = nn.Conv2d(dim // 2, dim, 1)  # 定义一个大小为1x1的卷积层恢复到原始通道数dim

    def forward(self, x):
        attn1 = self.conv0(x)  # 使用conv0处理输入x得到attn1 torch.Size([1, 64, 64, 64])
        attn2 = self.conv_spatial(attn1)  # 使用conv_spatial处理attn1得到attn2 torch.Size([1, 64, 64, 64])
        attn1 = self.conv1(attn1)  # 使用conv1处理attn1 torch.Size([1, 32, 64, 64])
        attn2 = self.conv2(attn2)  # 使用conv2处理attn2 torch.Size([1, 32, 64, 64])

        attn = torch.cat([attn1, attn2], dim=1)  # 将attn1和attn2在通道维度上拼接起来  torch.Size([1, 64, 64, 64])

        avg_attn = torch.mean(attn, dim=1, keepdim=True)  # 计算attn在通道维度上的平均值，并保持输出维度  torch.Size([1, 1, 64, 64])
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)  # 计算attn在通道维度上的最大值，并保持输出维度 torch.Size([1, 1, 64, 64])
        agg = torch.cat([avg_attn, max_attn], dim=1)  # 将平均值和最大值在通道维度上拼接起来    torch.Size([1, 2, 64, 64])

        sig = self.conv_squeeze(agg).sigmoid()  # 使用conv_squeeze处理拼接后的结果并使用sigmoid激活函数  torch.Size([1, 2, 64, 64])

        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)  # 根据sig对attn1和attn2加权求和  torch.Size([1, 32, 64, 64])
        attn = self.conv(attn)  # 使用conv处理加权后的结果            torch.Size([1, 64, 64, 64])
        return x * attn  # 返回输入x与处理后的attn相乘的结果

if __name__ == '__main__':
    block = Large_Selective_Kernel_Network(64)
    input = torch.rand(1, 64, 64, 64)
    output = block(input)
    print(input.size())  # 打印输入张量尺寸
    print(output.size())  # 打印输出张量尺寸

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")