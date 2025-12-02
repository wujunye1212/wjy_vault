import torch
import torch.nn as nn
from einops import rearrange

"""
    论文地址：https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Learning_a_Sparse_Transformer_Network_for_Effective_Image_Deraining_CVPR_2023_paper.pdf
    论文题目：Learning A Sparse Transformer Network for Effective Image Deraining（CVPR 2023）
    中文题目：学习稀疏的变压器网络以实现有效的图像去雨 （CVPR 2023）
    讲解视频：https://www.bilibili.com/video/BV1keBPYSEPi/
        Top-K 稀疏注意力（Top-k Sparse Attention ,TKSA）：
            设计目的：Transformer标准自注意力机制在计算时会考虑所有Q-K关系，这可能导致在图像恢复任务中引入无关特征的噪声干扰。
            理论支撑：通过保留最有用的自注意力值，避免无关信息在特征交互过程中的干扰，有助于更好地聚合特征。
"""

class TopK_Sparse_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(TopK_Sparse_Attention, self).__init__()
        self.num_heads = num_heads  # 注意力头的数量

        # 定义温度参数用于缩放注意力分数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 定义1x1卷积用于生成查询、键和值
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 定义深度可分离卷积
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 定义输出投影层
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # 定义注意力的dropout层
        self.attn_drop = nn.Dropout(0.)

        # 定义可学习的注意力权重
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入的形状

        qkv = self.qkv_dwconv(self.qkv(x))  # 计算查询、键和值
        q, k, v = qkv.chunk(3, dim=1)  # 将qkv分为查询、键和值

        # 重新排列查询、键和值的形状以适应多头注意力
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 对查询和键进行归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape  # 获取每个头的通道数

        # 初始化不同的掩码
        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # 根据注意力分数选择前k个元素进行计算
        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        # 对每个注意力分数进行softmax
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        # 计算输出
        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        # 根据可学习的权重组合输出
        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        # 重新排列输出的形状
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 应用输出投影层
        out = self.project_out(out)
        return out

if __name__ == '__main__':
    # 创建一个随机输入张量作为示例
    input_tensor = torch.randn(1, 64, 32, 32)  # 假设输入尺寸为 [batch_size, channels, height, width]

    # 实例化 TopK_Sparse_Attention 模块
    mdcr = TopK_Sparse_Attention(dim=64, num_heads=8, bias=True)

    # 将输入张量传递给 TopK_Sparse_Attention 模块并获取输出
    output_tensor = mdcr(input_tensor)

    # 打印输入和输出的形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")