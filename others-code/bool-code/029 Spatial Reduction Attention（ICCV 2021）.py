import torch
import torch.nn as nn
'''
    论文地址：https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Pyramid_Vision_Transformer_A_Versatile_Backbone_for_Dense_Prediction_Without_ICCV_2021_paper.html
    论文题目：Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions (ICCV 2021)
    中文题目：金字塔视觉变换器：无需卷积的多功能密集预测主干网络
    讲解视频：https://www.bilibili.com/video/BV1A5fyYEEN7/
    空间缩减自注意力（Spatial Reduction Attention）：
        在注意力操作之前的缩小了K和V的空间尺度，这大大减少了计算/内存开销。
'''
class SRAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        # 确保维度dim能够被头数num_heads整除
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim  # 输入特征的维度
        self.num_heads = num_heads  # 注意力头的数量
        head_dim = dim // num_heads  # 每个注意力头的维度
        # 设置缩放比例，如果提供了qk_scale就用它，否则使用默认值head_dim ** -0.5
        self.scale = qk_scale or head_dim ** -0.5

        # 定义查询（Query）的线性变换层
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # 定义键值对（Key-Value）的线性变换层，输出是两倍的dim，因为K和V共享同一个线性层
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # 定义注意力dropout层
        self.attn_drop = nn.Dropout(attn_drop)
        # 定义投影层，将多头注意力输出映射回原维度
        self.proj = nn.Linear(dim, dim)
        # 定义投影后的dropout层
        self.proj_drop = nn.Dropout(proj_drop)

        # 定义空间缩减率sr_ratio
        self.sr_ratio = sr_ratio
        # 如果空间缩减率大于1，则需要进行空间缩减操作，并且添加LayerNorm归一化
        if sr_ratio > 1:
            # 使用卷积层来进行空间缩减
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            # 添加LayerNorm层
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape  # 获取批次大小B、序列长度N以及通道数C
        # 对Q进行线性变换并重新排列为多头形式
        """
             self.q(x)                                                                          :  torch.Size([1, 256, 64])
             self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)                       :  torch.Size([1, 256, 8, 8])
             self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) ：torch.Size([1, 8, 256, 8])
        """
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 根据sr_ratio是否大于1来决定是否执行空间缩减
        if self.sr_ratio > 1:
            # 将输入转换为适合卷积层的格式
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # 应用空间缩减卷积 2D ---B, C, H, W -> B, C, (H, W) -> B, (H, W),C
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # 归一化处理
            x_ = self.norm(x_)  # torch.Size([1, 64, 64])
            # 对经过空间缩减后的数据进行键值对的线性变换，并重排为多头形式
            # self.kv(x_)                                                       :torch.Size([1, 64, 128])
            # self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads): torch.Size([1, 64, 2, 8, 8])
            # kv                                                                :torch.Size([2, 1, 8, 64, 8])
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            # 不进行空间缩减时直接对原始数据做键值对的线性变换，并重排为多头形式
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 应用softmax函数以确保注意力权重之和为1
        attn = attn.softmax(dim=-1)
        # 应用注意力dropout
        attn = self.attn_drop(attn)

        # 应用注意力权重到值V上，然后调整形状
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 应用最终的投影层
        x = self.proj(x)
        # 应用投影后的dropout
        x = self.proj_drop(x)
        # 返回最终结果
        return x

if __name__ == '__main__':

    dim = 64  # 输入通道数
    num_heads = 8  # 注意力头数量
    H, W = 16, 16  # 输入特征图的高度和宽度
    sr_ratio = 2  # 降采样比例

    # 创建SRAttention模块
    sr_attention = SRAttention(dim, num_heads, sr_ratio=sr_ratio)
    # 创建输入张量

    input = torch.randn(1, H * W, dim)
    output = sr_attention(input, H, W)
    print('input_size:',input.size())
    print('output_size:',output.size())

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")