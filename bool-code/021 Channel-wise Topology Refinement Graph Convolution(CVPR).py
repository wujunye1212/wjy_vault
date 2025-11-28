import torch
import torch.nn as nn
"""
论文地址：https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Channel-Wise_Topology_Refinement_Graph_Convolution_for_Skeleton-Based_Action_Recognition_ICCV_2021_paper.pdf
论文题目：Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition (CVPR 2021)
中文题目：用于基于骨架的动作识别的通道拓扑细化图卷积
讲解视频：https://www.bilibili.com/video/BV1pywmeKEs5/
        通道拓扑细化图卷积（CTR-GC）:
            以动态学习不同的拓扑结构，并有效地聚合不同通道中的联合特征，以进行基于骨架的动作识别。
            CTR-GC通过学习共享拓扑作为所有通道的通用先验，并使用每个通道的特定于通道的相关性对其进行优化，从而对通道拓扑进行建模。
            引入了很少的额外参数，并显着降低了通道拓扑建模的难度。


    相关通道注意力讲解：
        https://www.bilibili.com/video/BV1wFbAekEZG/
        https://www.bilibili.com/video/BV1fVxueLErc/
        https://www.bilibili.com/video/BV1aBxse5EEr/
"""
class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(CTRGC, self).__init__()

        # 保存输入和输出通道数
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 特殊情况处理：如果输入通道数为3或9，则设置固定的相对通道数和中间通道数
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
        else:
            # 对于其他情况，根据给定的比例计算相对通道数和中间通道数
            self.rel_channels = in_channels // rel_reduction

        # 定义四个卷积层
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)

        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

        # 定义tanh激活函数
        self.tanh = nn.Tanh()

    # 前向传播方法
    def forward(self, x, A=None, alpha=1):
        # ============= 获得通道间的图拓扑结构 ==================
        # 对输入x应用conv1, conv2, 和conv3，然后对conv1和conv2的结果沿着倒数第二个维度求平均
        x1 = self.conv1(x).mean(-2)     # torch.Size([32, 64, 9, 9]) ===> torch.Size([32, 8, 9])
        x2 = self.conv2(x).mean(-2)     # torch.Size([32, 64, 9, 9]) ===> torch.Size([32, 8, 9])

        # 计算x1与x2之间的差值并使用tanh激活函数
        """`unsqueeze` 函数用于在指定的位置插入一个维度（大小为1的维度）。
                当两个张量进行操作时，PyTorch 会自动应用广播规则来匹配形状。
                在这个例子中，`[N, C, 1]` 和 `[N, 1, C]` 相减会产生一个形状为 `[N, C, C]` 的张量,可以被看作是对 `x1` 和 `x2` 进行某种形式的关系建模，
                                                            它生成了一个矩阵，该矩阵中的每个元素代表了 `x1` 中对应位置与 `x2` 中所有位置之间的差值。
                之后，通过 `tanh` 激活函数将这些差值映射到一个特定的范围内（通常是 -1 到 1 之间），有助于后续的特征学习或注意力机制的构建。
                这种做法常见于图神经网络或关系型数据处理中，用以捕捉节点之间的相对关系或依赖性。
        """
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))     # torch.Size([32, 8, 9, 1]) - torch.Size([32, 8, 1, 9]) =  torch.Size([32, 8, 9, 9])
        # x1 = MLP(x1.unsqueeze(-1) + x2.unsqueeze(-2))         # 改进1 https://www.bilibili.com/video/BV1ENs9eiE2Z/

        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)      # torch.Size([32, 64, 9, 9])
        # =================================================

        # =============== 残差网络 ====================
        x3 = self.conv3(x)              # torch.Size([32, 64, 9, 9]) ===> torch.Size([32, 64, 9, 9])
        # =================================================

        # 使用einsum执行类似于矩阵乘法的操作，但它是沿着特定的维度进行的。
        ## x1 可以看作是对 x3 中各个元素的权重或注意力分数【共享图拓扑】
        # 具体来说，对于每个批次和每个通道，它执行了 U x V 矩阵（来自x1）和 T x V 矩阵（来自x3）之间的矩阵乘法，结果是一个 T x U 矩阵。
        ## 在图神经网络（GNNs）或时序数据处理中很常见，能够有效地结合空间关系（由 x1 编码）和特征信息（由 x3 提供），可用来模拟复杂的关系建模或注意力机制。
        x1 = torch.einsum('bcuv,bctv->bctu', x1, x3)            # torch.Size([32, 64, 9, 9])   torch.Size([32, 64, 9, 9]) ==> torch.Size([32, 64, 9, 9])

        return x1

if __name__ == '__main__':

    # 创建CTRGC实例，指定输入和输出通道数均为64、可以【降维】
    # block = CTRGC(in_channels=64, out_channels=32)
    # 创建CTRGC实例，指定输入和输出通道数均为64、可以【不变】
    block = CTRGC(in_channels=64, out_channels=64)
    # 创建CTRGC实例，指定输入和输出通道数均为64、可以【升维】
    # block = CTRGC(in_channels=64, out_channels=128)

    # 创建随机输入张量，形状为(32, 64, 9, 9)，其中32是批次大小，64是通道数，9x9是空间尺寸
    input = torch.rand(32, 64, 9, 9)

    # 将输入传递给CTRGC模块得到输出
    output = block(input)

    print(input.size())
    print(output.size())

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")