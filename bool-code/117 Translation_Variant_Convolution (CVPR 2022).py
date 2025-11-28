import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/abs/2203.10489
    论文题目：TVConv: Efﬁcient Translation Variant Convolution for Layout-aware Visual Processing(CVPR 2022)
    中文题目：TVConv：用于布局感知视觉处理的高效平移可变卷积 (CVPR 2022)
    讲解视频：https://www.bilibili.com/video/BV16PA6eVEYn/
        平移可变卷积（Translation Variant Convolution，TVConv）：
            实际意义：传统卷积（如普通卷积和深度卷积）具有平移等变性，无法适应图像中不同位置的特征。像素级动态卷积虽考虑了空间适应性，但存在内存和计算开销大的问题。
            实现方式：TVConv通过学习紧凑关联矩阵（affinity maps）来捕捉输入中像素对之间的关系，并利用这些信息生成可变权重，从而实现对特定图像布局的高度适应性，
                    避免了过度参数化导致的计算开销问题。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class _ConvBlock(nn.Sequential):
    """
    _ConvBlock类定义了一个简单的卷积块，包含卷积层、层归一化和ReLU激活函数。
    """
    def __init__(self, in_planes, out_planes, h, w, kernel_size=3, stride=1, bias=False):
        # 计算填充大小，使得输出大小与输入大小相同
        padding = (kernel_size - 1) // 2
        super(_ConvBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias),  # 卷积层
            nn.LayerNorm([out_planes, h, w]),  # 层归一化
            nn.ReLU(inplace=True)  # ReLU激活函数
        )

class TVConv(nn.Module):
    """
    TVConv类定义了一个基于位置映射的空间变体卷积模块。
    """
    def __init__(self,
                 channels,
                 TVConv_k=3,
                 stride=1,
                 TVConv_posi_chans=4,
                 TVConv_inter_chans=64,
                 TVConv_inter_layers=3,
                 TVConv_Bias=False,
                 h=3,
                 w=3,
                 **kwargs):
        super(TVConv, self).__init__()

        # 注册缓冲区变量，表示卷积核大小、步长、通道数等
        self.register_buffer("TVConv_k", torch.as_tensor(TVConv_k))
        self.register_buffer("TVConv_k_square", torch.as_tensor(TVConv_k**2))
        self.register_buffer("stride", torch.as_tensor(stride))
        self.register_buffer("channels", torch.as_tensor(channels))
        self.register_buffer("h", torch.as_tensor(h))
        self.register_buffer("w", torch.as_tensor(w))

        self.bias_layers = None

        # 计算输出通道数
        out_chans = self.TVConv_k_square * self.channels

        # 初始化位置映射参数
        self.posi_map = nn.Parameter(torch.Tensor(1, TVConv_posi_chans, h, w))
        nn.init.ones_(self.posi_map)  # 用1初始化

        # 创建权重层和偏置层
        self.weight_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, out_chans, TVConv_inter_layers, h, w)
        if TVConv_Bias:
            self.bias_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, channels, TVConv_inter_layers, h, w)

        # 初始化 Unfold 模块，用于提取局部区域
        self.unfold = nn.Unfold(TVConv_k, 1, (TVConv_k-1)//2, stride)

    def _make_layers(self, in_chans, inter_chans, out_chans, num_inter_layers, h, w):
        """
        创建卷积层序列。
        """
        layers = [_ConvBlock(in_chans, inter_chans, h, w, bias=False)]
        for i in range(num_inter_layers):
            layers.append(_ConvBlock(inter_chans, inter_chans, h, w, bias=False))
        layers.append(nn.Conv2d(
            in_channels=inter_chans,
            out_channels=out_chans,
            kernel_size=3,
            padding=1,
            bias=False))  # 最后一层卷积
        return nn.Sequential(*layers)

    def forward(self, x):
        # 计算卷积权重
        weight = self.weight_layers(self.posi_map)
        weight = weight.view(1, self.channels, self.TVConv_k_square, self.h, self.w) # torch.Size([1, 64, 9, 32, 32])
        # 利用 Unfold 模块获取局部区域，并按照权重进行加权求和
        out = self.unfold(x).view(x.shape[0], self.channels, self.TVConv_k_square, self.h, self.w) # torch.Size([2, 64, 9, 32, 32])
        """
            weight * out：对这两个张量在 TVConv_k_square 维度上进行逐元素相乘。这个操作相当于对每个位置的局部区域应用一个位置特定的卷积核。
            .sum(dim=2) ：在TVConv_k_square维度上对乘积结果进行求和。TVConv_k_square 代表卷积核的展开大小（即核的面积），
                所以这个求和操作相当于对每个局部区域的卷积结果进行加权求和，类似于传统卷积操作。        
        """
        out = (weight * out).sum(dim=2) #实现了基于位置的加权卷积操作，生成了一个新的特征图。 # torch.Size([2, 64, 32, 32])
        if self.bias_layers is not None:
            # 如果使用偏置，则加上偏置
            bias = self.bias_layers(self.posi_map)
            out = out + bias
        return out

if __name__ == "__main__":
    # 生成随机的输入和位置映射
    input = torch.rand(2, 64, 32, 32)  # 输入张量为NCHW格式
    # 创建TVConv模块
    model = TVConv(64, h=32, w=32)
    # 运行TVConv模块
    output = model(input)
    print('input_size:', input.size())  # 打印输入尺寸
    print('output_size:', output.size())  # 打印输出尺寸
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')
    print("公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")