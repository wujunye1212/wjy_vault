import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块

"""
论文地址：https://openreview.net/pdf?id=XhYWgjqCrV
论文题目：MOGANET: MULTI-ORDER GATED AGGREGATION NETWORK（ICLR 2024）
讲解视频：https://www.bilibili.com/video/BV1VRtLe8EAg
"""

class ElementScale(nn.Module):
    """
        `ElementScale` 类定义了一个可学习的逐元素缩放器。
        类似于自注意力机制中的权重分配，通过学习合适的缩放参数给不同的元素分配不同的权重，自动决定哪些特征应该被放大或者抑制。
        1. 特征调节：在网络的某个层之后使用 `ElementScale` 可以对特征图进行缩放，这有助于调整特定通道或位置的特征的重要性。
        2. 残差连接中的缩放：在构建残差网络时，有时需要将跳跃连接（skip connection）与主路径输出相加。
                在这种情况下，使用 `ElementScale` 可以帮助调整跳跃连接的强度，使得加法操作更加有效。
        3. 增强模型表达能力：通过引入额外的可学习参数，模型能够更好地拟合训练数据，并可能提高泛化性能。
    """

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()  # 调用父类初始化方法

        # 创建一个可学习的缩放因子self.scale，它将在前向传播时与输入张量逐元素相乘，从而实现对输入张量的逐元素缩放。【加权，权重调整，多阶通道XXX】
        # 如果embed_dims = 64 并且init_value = 0.5，那么self.scale将初始化为一个形状为(1, 64, 1, 1)的张量，其中所有元素都设置为0.5。
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),  # 初始缩放值
            requires_grad=requires_grad  # 指定是否需要计算梯度
        )

    def forward(self, x):
        return x * self.scale  # 返回输入张量与缩放因子相乘的结果

class ChannelAggregationFFN(nn.Module):  # 定义通道聚合前馈网络类
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): 特征维度。
        ffn_ratio : 提升隐藏层维度的倍率
        kernel_size (int): 深度卷积核大小。默认为3。
        ffn_drop (float, optional): 在前馈网络中元素被置零的概率。默认为0.0。【扔多少】
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 kernel_size=3,
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()  # 调用父类初始化方法

        self.embed_dims = embed_dims  # 设置特征维度

        # 计算前馈网络的隐藏层维度
        feedforward_channels = int(embed_dims * ffn_ratio)
        self.feedforward_channels = feedforward_channels  # 设置隐藏层维度

        # 定义第一个线性变换（卷积操作）
        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,  # 输入通道数
            out_channels=self.feedforward_channels,  # 输出通道数
            kernel_size=1        # 卷积核大小
        )

        # 定义深度卷积
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,  # 输入通道数
            out_channels=self.feedforward_channels,  # 输出通道数
            kernel_size=kernel_size,  # 卷积核大小   kernel_size=3
            stride=1,  # 步长
            padding=kernel_size // 2,  # 填充
            bias=True,  # 使用偏置
            groups=self.feedforward_channels  # 分组数，实现深度卷积
        )

        self.act = nn.GELU()  # 定义激活函数

        # 定义第二个线性变换（卷积操作）
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,  # 输入通道数
            out_channels=embed_dims,  # 输出通道数
            kernel_size=1  # 卷积核大小
        )

        self.drop = nn.Dropout(ffn_drop)  # 定义Dropout层

        # 定义分解卷积，将C个通道降维到1个通道
        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1,  # 输出通道数
            kernel_size=1  # 卷积核大小
        )

        # 定义ElementScale实例，用于特征分解时的缩放
        self.sigma = ElementScale(
            self.feedforward_channels,  # 通道数
            init_value=1e-5,  # 初始值
            requires_grad=True  # 是否需要计算梯度
        )

        self.decompose_act = nn.GELU()  # 定义分解后的激活函数

    # 关键
    def feat_decompose(self, x):

        Temp = self.decompose(x)            # [B, C, H, W] -> [B, 1, H, W] ：从多通道特征图中提取出一个单一通道的表示，可能是为了捕捉一些全局的信息或者主要的结构特征
        Temp = self.decompose_act(Temp)     # 对得到的单通道特征图 Temp 应用 GELU 激活函数。为了增加非线性，并且可能帮助模型学习更加复杂的模式
        Temp = x - Temp                     # 原始特征图x减去经过上述两步处理后的特征图Temp。这个减法操作可以被解释为去除或减弱了x中与Temp相似的部分
                                            # 如果Temp表示的是某种全局或主要的特征，那么x - Temp将保留那些不被Temp所代表的局部细节或差异信息
        Temp = self.sigma(Temp)             # self.sigma 本身是一个可学习的参数，可以根据训练过程中学到的信息来调整不同位置的缩放比例
        x = x + Temp                        # 将原始特征图 x 与【适当缩放的特定信息】相加。
        return x

    def forward(self, x):
        # 第一次投影
        x = self.fc1(x)  # 应用第一个线性变换
        x = self.dwconv(x)  # 应用深度卷积
        x = self.act(x)  # 应用激活函数
        x = self.drop(x)  # 应用Dropout

        # 特征分解
        x = self.feat_decompose(x)  # 应用特征分解

        # 第二次投影
        x = self.fc2(x)  # 应用第二个线性变换
        x = self.drop(x)  # 再次应用Dropout
        return x  # 返回最终输出


if __name__ == '__main__':
    input = torch.randn(1, 64, 32, 32)  # 创建随机输入张量

    CA = ChannelAggregationFFN(64)  # 实例化ChannelAggregationFFN对象

    output = CA(input)  # 通过模型传递输入

    print(' CA_input_size:', input.size())  # 打印输入尺寸
    print(' CA_output_size:', output.size())  # 打印输出尺寸

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")