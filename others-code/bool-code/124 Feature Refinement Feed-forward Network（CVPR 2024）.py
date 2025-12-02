import torch
import torch.nn as nn
from einops import rearrange, repeat
import math

"""
    论文地址：https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf
    论文题目：Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Reﬁnement for Image Restoration（CVPR 2024）
    中文题目：适者生存：用于图像恢复的具有注意力特征优化的自适应稀疏Transformer 
    讲解视频：https://www.bilibili.com/video/BV1DF9sYtEQr/
        特征细化前馈网络（Feature Refinement Feed-forward Network , FRFN）：
            实际意义：①解决特征图通道冗余：对所有通道应用相同的特征变换，会导致大量冗余信息，干扰模型对重要特征的学习。
                    ②增强特征表示：通过强化有用特征，使模型能更好地聚焦于关键信息，提升特征的可判别性。
            实现方式：先对输入特征进行 PConv操作筛选关键信息，经线性层突出有用信息，再通过特征通道切片以及后续操作，对其中一部分筛选冗余，最后再通过线性层。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class FRFN(nn.Module):
    def __init__(self, dim=16, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        # 定义第一个线性层，输出维度为 hidden_dim 的两倍，并使用激活函数
        self.linear1 = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),  # 线性层，输入维度为 dim，输出维度为 hidden_dim * 2
            act_layer()  # 激活函数，默认为 GELU
        )
        # 定义深度可分离卷积层（深度卷积），用于空间特征提取
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),  # 深度卷积
            act_layer()  # 激活函数
        )
        # 定义第二个线性层，将特征维度映射回 dim
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_dim, dim)  # 线性层，输入维度为 hidden_dim，输出维度为 dim
        )
        self.dim = dim  # 输入特征维度
        self.hidden_dim = hidden_dim  # 隐藏层维度

        # 定义部分卷积的通道分割
        self.dim_conv = self.dim // 4  # 部分卷积处理的通道数（四分之一的 dim）
        self.dim_untouched = self.dim - self.dim_conv  # 剩余未处理的通道数
        # 定义部分卷积操作，仅作用于 dim_conv 通道
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)  # 3x3 卷积，无偏置

    def forward(self, x):
        """
            输入：x 的形状为 (batch_size, hw, c)，其中 hw 为空间维度展开后的大小，c 为通道数
        """
        bs, hw, c = x.size()  # 获取输入的批量大小、空间维度和通道数
        hh = int(math.sqrt(hw))  # 计算输入的高度（假设输入是正方形）

        # 将输入从 3D 转换为 4D，恢复空间维度
        x = rearrange(x, 'b (h w) (c) -> b c h w', h=hh, w=hh)  # 调整形状为 (batch_size, c, h, w)
        # 对通道进行分割，x1 是部分卷积处理的通道，x2 是未处理的通道
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)  # 按通道分割

        # 对 x1 应用部分卷积
        x1 = self.partial_conv3(x1)  # 部分卷积操作
        # 将处理后的 x1 和未处理的 x2 拼接
        x = torch.cat((x1, x2), 1)  # 在通道维度上拼接
        # 将 4D 特征展平为 3D
        x = rearrange(x, 'b c h w -> b (h w) c', h=hh, w=hh)  # 调整形状为 (batch_size, hw, c)

        # 通过第一个线性层
        x = self.linear1(x)  # 线性映射，输出形状为 (batch_size, hw, hidden_dim * 2)

        # 使用门控机制对特征分割为两部分
        x_1, x_2 = x.chunk(2, dim=-1)  # 按最后一个维度分割为两部分
        # 将 x_1 恢复为 4D，用于深度卷积
        x_1 = rearrange(x_1, 'b (h w) (c) -> b c h w', h=hh, w=hh)  # 调整形状为 (batch_size, hidden_dim, h, w)
        x_1 = self.dwconv(x_1)  # 通过深度卷积提取空间特征
        x_1 = rearrange(x_1, 'b c h w -> b (h w) c', h=hh, w=hh)  # 再次展平为 3D
        # 使用门控机制，将深度卷积后的特征与 x_2 相乘
        x = x_1 * x_2  # 按元素相乘

        # 通过第二个线性层
        x = self.linear2(x)  # 线性映射回原始维度

        return x  # 返回输出

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

if __name__ == '__main__':
    H, W = 32, 32
    input1 = torch.randn(1, 64, H, W)  #
    model = FRFN(dim=64)
    input = to_3d(input1)
    output = model(input)
    output = to_4d(output,H,W)
    print('input_size:', input1.size())
    print('output_size:', output.size())
