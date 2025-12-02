import torch
import torch.nn as nn
import torch.nn.functional as F

"""
论文地址：https://openreview.net/pdf?id=XhYWgjqCrV
论文题目：MOGANET: MULTI-ORDER GATED AGGREGATION NETWORK（ICLR 2024）
讲解视频：https://www.bilibili.com/video/BV1RwtReMEzR/

            通道级特征融合 
"""

class ElementScale(nn.Module):
    """
        https://www.bilibili.com/video/BV1bztveGEz2/
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

class MultiOrderDWConv(nn.Module):
    """使用膨胀深度卷积核的多阶特征。

    参数:
        embed_dims (int): 输入通道数。
        dw_dilation (list): 三个深度卷积层的膨胀率。
        channel_split (list): 三个分割通道的比例。
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,],
                ):
        super(MultiOrderDWConv, self).__init__()

        # 计算各部分通道比例
        self.split_ratio = [i / sum(channel_split) for i in channel_split]  # 1/8   3/8   4/8

        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)       # 3/8 * embed_dims

        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)       # 4/8 * embed_dims

        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2      # 1/8  * embed_dims

        self.embed_dims = embed_dims

        assert len(dw_dilation) == len(channel_split) == 3  # 确保长度正确
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3  # 检查膨胀率范围
        assert embed_dims % sum(channel_split) == 0  # 确保embed_dims可以整除channel_split总和

        # 基础深度卷积
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,  # 根据膨胀率计算填充
            groups=self.embed_dims,                 # 分组数量等于输入通道数
            stride=1,                      # 设置步长
            dilation=dw_dilation[0],      # 膨胀率  1
        )
        # 第二个深度卷积
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1,
            dilation=dw_dilation[1],        # 膨胀率  2
        )

        # 第三个深度卷积
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1,
            dilation=dw_dilation[2],        # 膨胀率  3
        )

        # 逐点卷积
        self.PW_conv = nn.Conv2d(  # 点卷积用于融合不同分支
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1
        )


    def forward(self, x):
        x_0 = self.DW_conv0(x)  # 第一个 5X5深度卷积

                            #[:,1/8  * embed_dims ：1/8  * embed_dims+ 3/8 * embed_dims: ...]
        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])

                            # [:, embed_dims- 4/8 * embed_dims ：embed_dims: ...]
        x_2 = self.DW_conv2(x_0[:, self.embed_dims-self.embed_dims_2:, ...])

        x = torch.cat([x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)  # 按通道维度拼接

        x = self.PW_conv(x)     # 点卷积用于融合不同分支
        return x

class MultiOrderGatedAggregation(nn.Module):
    """
        具有多阶门控聚合的空间块
    参数:
        embed_dims (int): 输入通道数。
        attn_dw_dilation (list): 三个深度卷积层的膨胀率。
        attn_channel_split (list): 分割通道的比例。
        attn_act_type (str): 空间块的激活类型，默认为'SiLU'。
    """
    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_force_fp32=False,
                 ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )

        self.proj_2 = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # 门控和值激活
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()

        # 分解操作
        self.sigma = ElementScale(embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):

        x = self.proj_1(x)

        # 对x进行全局平均池化
        x_d = F.adaptive_avg_pool2d(x, output_size=1)

        x = x + self.sigma(x - x_d)  # 加上分解后的调整

        x = self.act_value(x)  # 应用激活函数 这里与文章不对应 可以自行修改

        return x

    def forward(self, x):
        shortcut = x.clone()

        # 图中蓝色框
        x = self.feat_decompose(x)

        # 图中灰色框  左F 右G
        F = self.gate(x)
        G = self.value(x)

        # 两个SiLU和点乘
        x = self.proj_2(self.act_gate(F) * self.act_gate(G))

        # 输出
        x = x + shortcut  # 残差连接

        return x

if __name__ == '__main__':

    input = torch.randn(1, 64, 32, 32)

    MOGA = MultiOrderGatedAggregation(64)

    output = MOGA(input)
    print(' MOGA_input_size:', input.size())
    print(' MOGA_output_size:', output.size())

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")