import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_ as trunc_normal_init

'''
    论文地址：https://arxiv.org/pdf/2312.17071
    论文题目：SCTNet: Single-Branch CNN with Transformer Semantic Information for Real-Time Segmentation （AAAI 2024）
    中文题目：SCTNet：具有 Transformer 语义信息的单分支 CNN，用于实时分割（AAAI 2024）
    讲解视频：https://www.bilibili.com/video/BV15SBsYcENj/
        条纹卷积注意力（Convolutional Attention）：
            优点：缓解CNN与Transformer特征之间的语义差距，仅使用卷积操作就可以捕获长距离上下文。
            特点：考虑到效率问题，将采用条纹卷积来实现卷积注意力，即利用1×k和k×1卷积来近似一个k×k卷积层。
'''
class ConvolutionalAttention(nn.Module):
    def __init__(self,
                 in_channels,  # 输入通道数
                 inter_channels,  # 中间通道数
                 num_heads=8):  # 注意力头数，默认值为8
        super(ConvolutionalAttention, self).__init__()
        out_channels = in_channels  # 输出通道数等于输入通道数
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.inter_channels = inter_channels  # 设置中间通道数
        self.num_heads = num_heads  # 设置注意力头数
        self.norm = nn.SyncBatchNorm(in_channels)  # 同步批量归一化

        # 初始化卷积核参数
        self.kv = nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1))
        self.kv3 = nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7))
        trunc_normal_init(self.kv, std=0.001)  # 使用截断正态分布初始化
        trunc_normal_init(self.kv3, std=0.001)

    def _act_dn(self, x):
        x_shape = x.shape  # 获取输入张量的形状，n,c_inter,h,w
        h, w = x_shape[2], x_shape[3]  # 高度和宽度
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])  # 重塑张量形状
        x = F.softmax(x, dim=3)  # 对最后一个维度应用softmax
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)  # 归一化处理
        x = x.reshape([x_shape[0], self.inter_channels, h, w])  # 恢复张量形状
        return x

    def forward(self, x):
        x = self.norm(x)  # 进行批量归一化  1, 64, 32, 32

        """
            self.kv  纵向卷积（高度方向）
            padding=(3, 0) 在高度方向上有填充，而宽度方向上没有填充
        """
        x1 = F.conv2d(x,self.kv,bias=None,stride=1,padding=(3,0)) # torch.Size([1, 32, 32, 32])
        x1 = self._act_dn(x1)  # 激活和归一化处理                       torch.Size([1, 32, 32, 32])
        x1 = F.conv2d(x1, self.kv.transpose(1, 0), bias=None, stride=1,padding=(3,0)) # torch.Size([1, 64, 32, 32])

        """
            self.kv3  横向卷积（宽度方向）
            padding=(0, 3) 在宽度方向上有填充，而高度方向上没有填充。
        """
        x3 = F.conv2d(x,self.kv3,bias=None,stride=1,padding=(0,3)) # torch.Size([1, 32, 32, 32])
        x3 = self._act_dn(x3)  # 激活和归一化处理 torch.Size([1, 32, 32, 32])
        x3 = F.conv2d(x3, self.kv3.transpose(1, 0), bias=None, stride=1, padding=(0,3)) # torch.Size([1, 64, 32, 32])

        x = x1 + x3  # 合并两个卷积的结果
        return x

if __name__ == "__main__":
    model = ConvolutionalAttention(64, 32, 8)  # 创建 ConvolutionalAttention 实例

    # 创建随机输入张量
    input = torch.randn(1, 64, 32, 32)  # 批量大小为1，输入通道数为64，尺寸为32x32

    # 通过模型进行前向传播
    output = model(input)

    # 打印输出张量的形状
    print("Output shape:", output.shape)
    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息


