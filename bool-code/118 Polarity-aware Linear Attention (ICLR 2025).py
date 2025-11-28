import torch
import torch.nn as nn
"""
    论文地址：https://arxiv.org/abs/2501.15061
    论文题目：PolaFormer: Polarity-aware Linear Attention for Vision Transformers (ICLR 2025)
    中文题目：PolaFormer：用于视觉Transformer的极性感知线性注意力机制  (ICLR 2025)
    讲解视频：https://www.bilibili.com/video/BV1RgPcefEUo/
        极性感知注意力（Polarity-Aware Attention）：
            实际意义：Transformer在视觉任务表现出色，但自注意力机制复杂度使其在处理长序列或高分辨率图像时计算开销大，
                    稀疏注意力虽降低计算成本，但牺牲上下文信息；线性注意力用核化特征图将复杂度降至线性，但存在表达能力不足问题。
            实现方式：通过分离正负QK对分别进行处理，有效恢复负向的交互信息，同时保持线性复杂度，并提出了一个简单可学习功率函数来实现减少信息熵的功能
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""
class PolaLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, kernel_size=5, alpha=4):
        super().__init__()
        # 确保维度能被头数整除
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        # 初始化基本参数
        self.dim = dim  # 输入特征维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = dim // num_heads  # 每个头的维度

        # 生成查询（q）和门控（g）向量的线性层，输出维度是2*dim
        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)

        # 生成键（k）和值（v）向量的线性层
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # 注意力dropout和投影层
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 最终投影层
        self.proj_drop = nn.Dropout(proj_drop)

        # 空间缩减相关参数
        self.sr_ratio = sr_ratio  # 缩减比例
        if sr_ratio > 1:
            # 使用卷积进行空间维度缩减（类似下采样）
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)  # 归一化层

        # 深度可分离卷积（Depthwise Convolution）
        self.dwc = nn.Conv2d(
            in_channels=self.head_dim,
            out_channels=self.head_dim,
            kernel_size=kernel_size,
            groups=self.head_dim,  # 组数等于输入通道数，实现深度可分离
            padding=kernel_size // 2  # 保持特征图尺寸不变
        )

        # 可学习参数：增强非线性表达
        ## 可学习功率函数的核心代码
        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha  # 缩放因子

        # 缩放因子和位置编码
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))

        self.kernel_function = nn.ReLU()

        # 打印配置信息
        print(f'Linear Attention sr_ratio{sr_ratio} f{alpha} kernel{kernel_size}')

    def forward(self, x, H, W):
        # 输入形状：B: batch大小, N: 序列长度, C: 通道数
        B, N, C = x.shape

        # 生成查询向量q和门控向量g
        # torch.Size([2, 64, 128]) ====》torch.Size([2, 64, 128])+torch.Size([2, 64, 128])
        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)  # 拆分为q和g

        # 处理键值对（根据sr_ratio决定是否进行空间缩减）
        if self.sr_ratio > 1:
            # 调整维度顺序以进行空间缩减卷积
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # 卷积+展平
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            # torch.Size([2, 64, 128]) ===> torch.Size([2, 2, 64, 128])
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)

        # torch.Size([2, 64, 128]) + torch.Size([2, 64, 128])
        k, v = kv[0], kv[1]  # 拆分出k和v
        n = k.shape[1]  # 键值对的序列长度  64
        # 添加位置编码
        k = k + self.positional_encoding # torch.Size([2, 64, 128])

        # 计算缩放因子和幂次增强参数
        scale = nn.Softplus()(self.scale)  # 保证缩放因子为正
        # 应用缩放
        q = q / scale
        k = k / scale

        # 调整维度为多头形式
        # q形状: [B, num_heads, N, head_dim] torch.Size([2, 8, 64, 16])
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        # k形状: [B, num_heads, n, head_dim]  torch.Size([2, 8, 64, 16])
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        # v形状: [B, num_heads, n, head_dim]  torch.Size([2, 8, 64, 16])
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        # 可学习功率函数的核心代码 ===>实现非线性增强
        ## 通过可学习的幂函数（learnable power function）对不同维度的注意力信号进行差异化缩放，从而降低注意力分布的信息熵
        power = 1 + self.alpha * torch.sigmoid(self.power)
        # 计算正负符号的查询Q
        q_pos = self.kernel_function(q) ** power   # 正Q torch.Size([2, 8, 64, 16])
        q_neg = self.kernel_function(-q) ** power  # 负Q torch.Size([2, 8, 64, 16])
        q_sim = torch.cat([q_pos, q_neg], dim=-1)  # 正负Q拼接 正方向
        q_opp = torch.cat([q_neg, q_pos], dim=-1)  # 负正Q拼接 反方向

        # 计算正负符号的键K
        k_pos = self.kernel_function(k) ** power   # 正K torch.Size([2, 8, 64, 16])
        k_neg = self.kernel_function(-k) ** power  # 负K torch.Size([2, 8, 64, 16])
        k = torch.cat([k_pos, k_neg], dim=-1)  # 正负K拼接

        # 将值向量分为两部分
        v1, v2 = torch.chunk(v, 2, dim=-1)  # 沿最后一个维度切分

        # 1、计算正相似性部分【+】
        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)  # 归一化因子
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))  # 键值交互
        x_sim = q_sim @ kv * z  # 相似性计算
        # 2、计算负相似性部分【-】
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z
        # 3、合并结果
        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C)  # 调整维度恢复形状

        # 空间缩减时的插值恢复
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(
                v.transpose(-2, -1).reshape(B * self.num_heads, -1, n),
                size=N,
                mode='linear'
            ).reshape(B, self.num_heads, -1, N).transpose(-2, -1)

        # 深度可分离卷积处理
        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)

        # 门控操作和最终输出
        x = x + v  # 残差连接
        x = x * g  # 应用门控
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

from einops import rearrange
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

if __name__ == '__main__':
    B = 2  # Batch大小
    C = 128  # 通道数
    H = 8  # 特征图高度
    W = 8  # 特征图宽度
    N = H * W  # 序列长度（如图像块数量） H *W

    # 定义一个4D张量
    tensor_4d = torch.randn(B, C, H, W) # torch.Size([2, 128, 8, 8])
    # 使用 to_3d 函数将4D张量转换为3D张量
    input_tensor = to_3d(tensor_4d) # torch.Size([2, 64, 128])
    # 初始化注意力模块
    block = PolaLinearAttention(dim=C, num_patches=N, sr_ratio=1)
    output = block(input_tensor, H, W)  # 前向传播
    output = to_4d(output,H,W)

    print("Input size:", tensor_4d.size())  # 输出输入尺寸
    print("Output size:", output.size())  # 输出结果尺寸