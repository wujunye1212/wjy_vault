from torch import nn
import torch
'''
    论文地址：https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html
    论文题目：Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks (CVPR 2023)
    中文题目：追求更高的 FLOPS，实现更快的神经网络
    讲解视频：https://www.bilibili.com/video/BV1GVyaY8Ezd/
    
    现有方法的局限性：现有方法在减少 FLOP 时存在内存访问增加、计算碎片化、对特定硬件依赖等问题，导致实际运行速度提升不明显。
    
    Partial Conv：
      本文提出新卷积神经网络——PConv，只使用了部分输入通道并减少了内存访问次数，同时，它们还能够在一定程度上保留通道之间的信息流动，使得后续的神经网络能够更好地处理数据。
      因此可以减少冗余计算和内存访问来更有效地提取空间特征，在各种设备上运行速度更快，而不会影响准确性。
'''
class Partial_conv(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div              # 计算将被3x3卷积处理的维度大小
        self.dim_untouched = dim - self.dim_conv3  # 计算不会被卷积操作影响的维度大小
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)  # 创建3x3卷积层，不带偏置

        # 根据参数forward选择前向传播方式
        if forward == 'slicing':
            self.forward = self.forward_slicing  # 使用切片方式
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat  # 使用分割和连接方式
        else:
            raise NotImplementedError  # 如果提供的forward不是预设的两种之一，则抛出未实现错误

    def forward_split_cat(self, x):
        # 既可用于训练也可用于推理
        """"
                self.dim_conv3      16
                self.dim_untouched  48
                x1                  torch.Size([1, 16, 224, 224])
                x2                  torch.Size([1, 48, 224, 224])
        """
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)  # 将输入沿通道维度分割成两部分
        x1 = self.partial_conv3(x1)  # 对第一部分进行卷积操作
        x = torch.cat((x1, x2), 1)  # 将卷积后的第一部分与未改变的第二部分在通道维度上拼接
        return x  # 返回最终拼接后的张量

    def forward_slicing(self, x):
        # 只适用于推理阶段
        x = x.clone()  # 复制输入以保持原始输入不变，用于后续残差连接
        # self.dim_conv3      16   [0~16]
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])  # 对选定维度【0：self.dim_conv3】应用卷积操作

        return x

if __name__ == '__main__':

    # model = Partial_conv(64, 4, 'split_cat')
    model = Partial_conv(64, 4, 'slicing')

    input_tensor = torch.randn(1, 64, 224, 224)

    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
