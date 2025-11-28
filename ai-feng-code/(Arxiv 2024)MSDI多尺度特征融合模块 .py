import torch
import torch.nn as nn
import torch.nn.functional as F
# https://github.com/yaoppeng/U-Net_v2/tree/master
# https://arxiv.org/pdf/2311.17791


'''
即插即用模块：MSDI多尺度特征融合模块      Arxiv 2024
在本文中，我们介绍了 U-Netv2，这是一种用于医学图像分割的新型强大且高效的 U-Net 变体。
它旨在增强语义信息对低级特征的注入，同时用更精细的细节来提炼高级特征。
对于输入图像，我们首先使用深度神经网络编码器提取多级特征。
接下来，我们通过注入来自更高级别特征的语义信息

并通过 Hadamard集成来自较低级别特征的更精细细节来增强每个级别的特征图。
我们新颖的跳跃连接为所有级别的功能提供了丰富的语义特征和复杂的细节。
改进的功能随后传输到解码器，以进行进一步处理和分割。
我们的方法可以无缝集成到任何编码器-解码器网络中.

适用于:本文用于医学图像分割，MSDI多尺度特征融合模块适用于所有CV2维任务
'''
class MSDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(c, channel[1], kernel_size=3, stride=1, padding=1) for c in channel])

    def forward(self, x):
        ans = torch.ones_like(x[1])
        target_size = x[1].shape[-2:]

        for i, x in enumerate(x):
            if x.shape[-1] > target_size[0]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[0]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)
        return ans
# 测试 MSDI 模块
if __name__ == '__main__':
    # 定义输入张量
    input1 = torch.randn(1, 16, 32, 32)
    input2 = torch.randn(1, 32, 64, 64)
    input3 = torch.randn(1, 64, 128, 128)
    input4 = torch.randn(1, 128, 256, 256)

    # 将输入张量放入列表
    # inputs = [input1, input2, input3]
    inputs = [input1, input2, input3,input4]

    # 定义 MSDI 模块，通道数为输入的通道数
    model = MSDI([16, 32, 64,128])

    # 执行前向传播
    output = model(inputs)

    # 打印输出张量的形状
    print(f"Output shape: {output.shape}")