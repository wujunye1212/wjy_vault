import torch
import torch.nn as nn
'''
    论文地址：https://arxiv.org/pdf/2303.16900
    论文题目：InceptionNeXt: When Inception Meets ConvNeXt (CVPR 2024)
    中文题目：InceptionNeXt：当Inception遇到ConvNeXt
    讲解视频：https://www.bilibili.com/video/BV1zNCDY9EMY/
    InceptionNeXt：
      将大型核深度卷积分解为四个沿通道维度的平行分支，即身份映射/正方形核/两个正交核，不仅享有较高的吞吐量而且保持竞争力的性能。
'''
class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        # 根据branch_ratio计算出分支卷积层的通道数gc
        gc = int(in_channels * branch_ratio)

        """
        kernel_size:指定了卷积核（或滤波器）的大小。它决定了每次卷积操作时覆盖输入张量的区域大小。
                    如果是一个整数，则表示方形卷积核；
                    如果是元组形式如(height, width)，则分别指定高度和宽度方向上的卷积核大小。
        padding    :在输入张量边缘周围添加额外的零值，以控制输出张量的空间维度。它可以避免由于卷积操作导致的边界信息丢失。
                    如果是一个整数，则在每个边界的四周均匀地添加相同数量的零；
                    如果是元组形式如(top, bottom, left, right)，则分别指定上、下、左、右四个方向上的填充大小。
                    对于二维卷积，常见的简化形式为(height_padding, width_padding)。
        """
        # 深度可分离卷积层，处理方形区域，使用gc个分组进行卷积
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)

        # 深度可分离卷积层，处理水平 W 方向上的条形区域，使用gc个分组进行卷积
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),groups=gc)
        # 深度可分离卷积层，处理竖直 H 方向上的条形区域，使用gc个分组进行卷积
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),groups=gc)

        # 计算用于拆分输入张量x的索引，以便将x分配到不同的卷积路径中
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    # 前向传播函数，定义了数据流经网络的方式
    def forward(self, x):
        # 使用预先计算好的索引split_indexes来拆分输入张量x，分别对应不经过卷积(身份ID)、方形卷积、宽条形卷积和高条形卷积的部分
        """
            torch.Size([1, 40, 224, 224])
            torch.Size([1, 8, 224, 224])
            torch.Size([1, 8, 224, 224])
            torch.Size([1, 8, 224, 224])
        """
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        # 将不同路径处理后的张量沿通道维度拼接起来，并作为最终输出返回
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )
# 身份映射/正方形核/两个正交核
if __name__ == '__main__':
    model = InceptionDWConv2d(in_channels=64)

    input_tensor = torch.randn(1, 64, 224, 224)

    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")