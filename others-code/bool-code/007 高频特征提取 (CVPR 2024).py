import torch
import torch.nn as nn

"""
论文地址：https://arxiv.org/abs/2404.13537
【高频特征提取】
论文题目：Bracketing Image Restoration and Enhancement with High-Low Frequency Decomposition（CVPR 2024）
讲解视频：https://www.bilibili.com/video/BV1BetCemEyy/

    高频信息代表图像的细节，因此需要采用较小的感受野来更好地关注局部图像信息，以实现图像细节恢复。

    为了处理高频信息，我们提出了局部特征提取块来进行特征提取，该模块由多个小卷积核的卷积层和密集连接组成。
    小卷积核能够更好地聚焦于细节区域，而残差连接则擅长探索高频信息[11, 24]。因此，这种组合对于提取高频信息非常有效。
"""
class Dense(nn.Module):
    def __init__(self, in_channels):
        super(Dense, self).__init__()

        """
            **小卷积核**：使用较小的卷积核（例如3x3或更小）有助于网络捕捉图像中的细微结构和纹理。这些细小的细节是高频信息的关键组成部分。
        """
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv6 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)

        self.gelu = nn.GELU()

    def forward(self, x):

        """
            密集连接：通过在不同层之间建立密集连接，网络可以将低级特征与高级特征相结合，从而增强对图像中细节部分的理解。
                            这有助于保持并强化高频信息。

            残差连接：残差学习机制允许网络直接学习输入与输出之间的差异，这对于处理高频信息特别有用，
                            因为它可以帮助网络专注于那些需要被加强或修复的细节部分。
        """

        # x : torch.Size([1, 32, 64, 64])
        x1 = self.conv1(x)      # torch.Size([1, 32, 64, 64])
        x1 = self.gelu(x1+x)    # torch.Size([1, 32, 64, 64])

        x2 = self.conv2(x1)      # torch.Size([1, 32, 64, 64])
        x2 = self.gelu(x2+x1+x)   # torch.Size([1, 32, 64, 64])

        x3 = self.conv3(x2)             # torch.Size([1, 32, 64, 64])
        x3 = self.gelu(x3+x2+x1+x)      # torch.Size([1, 32, 64, 64])

        x4 = self.conv4(x3)             # torch.Size([1, 32, 64, 64])
        x4 = self.gelu(x4+x3+x2+x1+x)   # torch.Size([1, 32, 64, 64])

        x5 = self.conv5(x4)                 # torch.Size([1, 32, 64, 64])
        x5 = self.gelu(x5+x4+x3+x2+x1+x)    # torch.Size([1, 32, 64, 64])

        x6 = self.conv6(x5)                 # torch.Size([1, 32, 64, 64])
        x6 = self.gelu(x6+x5+x4+x3+x2+x1+x) # torch.Size([1, 32, 64, 64])

        return x6

if __name__ == '__main__':

    # 实例化模型对象
    model = Dense(32)

    # 生成随机输入张量
    input = torch.randn(1, 32, 64, 64)

    # 执行前向传播
    output = model(input)
    print('input_size:',input.size())
    print('output_size:',output.size())


    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
