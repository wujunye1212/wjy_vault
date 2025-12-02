import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

"""
    论文地址：https://arxiv.org/pdf/2310.05026
    论文题目：Low-Resolution Self-Attention for Semantic Segmentation (TPAMI 2025)
    中文题目：用于语义分割的低分辨率自注意力机制(TPAMI 2025)
    讲解视频：https://www.bilibili.com/video/BV1pDsizHE79/
    低分辨率自注意力机制（Low-Resolution Self-Attention, LRSA）：
        实际意义：①高分辨率自注意力的计算复杂度高：传统Transformer的计算复杂度呈输入分辨率平方级，这会导致高分辨率输入时出现严重的计算开销和显存占用。
                ②全局上下文与局部细节难以兼顾：下采样注意力虽然对key和value 进行下采样以降低复杂度，但query仍保持高分辨率，计算量仍较大。
                ③受限于高分辨率依赖：普遍认为语义分割必须依赖高分辨率特征才能获得精确边界。
        实现方式：输入特征下采样在低分辨率上计算自注意力双线性插值还原尺寸
"""

class LRSA(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pooled_sizes=[11, 8, 6, 4], q_pooled_size=2, q_conv=False):
        super().__init__()

        # 确保通道数 dim 能被注意力头数 num_heads 整除
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        # 保存基本参数
        self.dim = dim                    # 输入特征维度（通道数）
        self.num_heads = num_heads        # 多头注意力的头数
        self.num_elements = np.array([t * t for t in pooled_sizes]).sum()  # 所有池化层输出特征的元素总数
        head_dim = dim // num_heads       # 每个注意力头的通道维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放系数（防止注意力分布过于尖锐）

        # 定义用于生成 Q（查询向量）的线性层
        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        # 定义用于生成 K（键向量）和 V（值向量）的线性层
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))

        # 注意力与输出的随机失活（Dropout）
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)   # 注意力输出后的线性投影
        self.proj_drop = nn.Dropout(proj_drop)

        # 定义多尺度池化尺寸（用于金字塔特征提取）
        self.pooled_sizes = pooled_sizes
        self.pools = nn.ModuleList()
        self.eps = 0.001                  # 防止除零的小常数

        # 对金字塔特征进行层归一化
        self.norm = nn.LayerNorm(dim)

        # Q 分支使用的自适应池化尺寸（决定是否下采样 Q）
        self.q_pooled_size = q_pooled_size

        # 定义一组深度可分离卷积，用于多尺度特征增强
        self.d_convs = nn.ModuleList([
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32) for _ in pooled_sizes
        ])

        # （可选）为 Q 分支添加深度卷积增强
        if q_conv and self.q_pooled_size > 1:
            self.q_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
            self.q_norm = nn.LayerNorm(dim)
        else:
            self.q_conv = None
            self.q_norm = None

    # ----------- 工具函数部分 -----------
    @staticmethod
    def to_3d(x):
        # 将输入从 [B, C, H, W] 转换为 [B, H*W, C]
        return rearrange(x, 'b c h w -> b (h w) c')

    @staticmethod
    def to_4d(x, h, w):
        # 将特征从 [B, H*W, C] 还原为 [B, C, H, W]
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

    # ----------- 前向传播部分 -----------
    def forward(self, x, H, W):
        # 将输入变为 3D 张量，方便后续矩阵运算
        x = self.to_3d(x)
        B, N, C = x.shape
        H, W = int(H), int(W)

        # =============== 构建 Q 查询特征 ===============
        if self.q_pooled_size > 1:
            # 自适应确定 Q 的池化尺寸（保持宽高比例一致）
            q_pooled_size = (self.q_pooled_size, round(W * float(self.q_pooled_size) / H + self.eps)) \
                if W >= H else (round(H * float(self.q_pooled_size) / W + self.eps), self.q_pooled_size)

            # 对输入特征进行自适应平均池化（降低空间分辨率）
            q = F.adaptive_avg_pool2d(x.transpose(1, 2).reshape(B, C, H, W), q_pooled_size)
            _, _, H1, W1 = q.shape

            # 如果启用了 Q 卷积增强，则卷积 + 归一化
            if self.q_conv is not None:
                q = q + self.q_conv(q)
                q = self.q_norm(q.view(B, C, -1).transpose(1, 2))
            else:
                q = q.view(B, C, -1).transpose(1, 2)

            # 线性变换生成 Q，拆分多头格式 [B, num_heads, N, head_dim]
            q = self.q(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        else:
            # 不进行池化，直接使用原特征生成 Q
            H1, W1 = H, W
            if self.q_conv is not None:
                x1 = x.view(B, -1, C).transpose(1, 2).reshape(B, C, H1, W1)
                q = x1 + self.q_conv(x1)
                q = self.q_norm(q.view(B, C, -1).transpose(1, 2))
                q = self.q(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            else:
                q = self.q(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # =============== 构建 K / V 特征（多尺度金字塔） ===============
        pools = []
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)

        # 针对每个尺度执行金字塔池化
        for (pooled_size, l) in zip(self.pooled_sizes, self.d_convs):
            # 计算自适应池化的尺寸
            pooled_size = (pooled_size, round(W * pooled_size / H + self.eps)) if W >= H else (
                round(H * pooled_size / W + self.eps), pooled_size)
            # 池化操作
            pool = F.adaptive_avg_pool2d(x_, pooled_size)
            # 卷积增强局部特征
            pool = pool + l(pool)
            # 展平后保存
            pools.append(pool.view(B, C, -1))

        # 拼接所有尺度特征并归一化
        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0, 2, 1))
        # 生成 K 和 V 特征（各一半）
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # =============== 多头自注意力计算 ===============
        # 计算注意力权重矩阵
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # 加权求和得到新的特征表示
        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, -1, C)

        # =============== 输出投影与恢复空间结构 ===============
        x = self.proj(x)
        if self.q_pooled_size > 1:
            # 将 Q 的低分辨率特征插值回原尺寸
            x = x.transpose(1, 2).reshape(B, C, H1, W1)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            x = x.view(B, C, -1).transpose(1, 2)

        # 恢复为 [B, C, H, W] 结构
        x = self.to_4d(x, H, W)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)  # 模拟输入图像 [batch=1, 通道=32, 高=50, 宽=50]
    attn = LRSA(dim=32, num_heads=4)  # 构建模块（输入通道=32，多头=4）
    out = attn(x, 50, 50)             # 前向计算
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {out.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")