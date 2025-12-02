import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_, DropPath

"""
    论文地址：https://arxiv.org/pdf/2412.14598
    论文题目：SparseViT: Nonsemantics-Centered, Parameter-Efficient Image Manipulation Localization through Spare-Coding Transformer (AAAI 2025)
    中文题目：SparseViT：通过稀疏编码变压器实现非语义中的高效图像操纵定位(AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV1TjcJeZEda/
        稀疏注意力（Sparse Self-Attention, SSA）
             发现问题：传统自注意力的全局交互模式在图像篡改定位中会引入大量无关信息。
             解决思路：通过引入 “稀疏率” 超参数，将特征图划分为不重叠的张量块，仅在相同颜色块内进行自注意力计算，抑制语义信息表达，专注于提取非语义特征，同时减少计算量。
    类似思路：https://www.bilibili.com/video/BV1keBPYSEPi/
"""

# 是否使用层缩放
layer_scale = True
# 初始化值
init_value = 1e-6

# 深度卷积类
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        # 定义深度卷积层
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # 调整输入张量的形状
        x = x.transpose(1, 2).view(B, C, H, W)
        # 应用深度卷积
        x = self.dwconv(x)
        # 调整输出张量的形状
        x = x.flatten(2).transpose(1, 2)
        return x

# 多层感知机类
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 定义第一个全连接层
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 定义深度卷积层
        self.dwconv = DWConv(hidden_features)
        # 激活层
        self.act = act_layer()
        # 定义第二个全连接层
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout层
        self.drop = nn.Dropout(drop)
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 初始化权重和偏置
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # 前向传播
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 块划分函数
def block(x, block_size):
    B, H, W, C = x.shape
    # 计算需要填充的高度和宽度
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    # 填充
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    # 重塑张量
    x = x.reshape(B, Hp // block_size, block_size, Wp // block_size, block_size, C)
    # 调整维度顺序
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x, H, Hp, C

# 块合并函数
def unblock(x, Ho):
    B, H, W, win_H, win_W, C = x.shape
    # 调整维度顺序并重塑
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H * win_H, W * win_W, C)
    Wp = Hp = H * win_H
    Wo = Ho
    # 截断多余的填充
    if Hp > Ho or Wp > Wo:
        x = x[:, :Ho, :Wo, :].contiguous()
    return x

# 稀疏化函数
def alter_unsparse(x, H, Hp, C, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    # 重塑张量
    x = x.reshape(-1, Hp // sparse_size, Hp // sparse_size, sparse_size, sparse_size, C)
    # 调整维度顺序
    x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
    # 合并块
    out = unblock(x, H)
    # 调整维度顺序
    out = out.permute(0, 3, 1, 2)
    return out

# 稀疏化函数
def alter_sparse(x, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    # 确保图像大小可被块大小整除
    assert x.shape[1] % sparse_size == 0 & x.shape[2] % sparse_size == 0, 'image size should be divisible by block_size'
    # 计算网格大小
    grid_size = x.shape[1] // sparse_size
    # 块划分
    out, H, Hp, C = block(x, grid_size)
    # 调整维度顺序
    out = out.permute(0, 3, 4, 1, 2, 5).contiguous()
    # 重塑张量
    out = out.reshape(-1, sparse_size, sparse_size, C)
    # 调整维度顺序
    out = out.permute(0, 3, 1, 2)
    return out, H, Hp, C

# 注意力机制类
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 计算缩放因子
        self.scale = qk_scale or head_dim ** -0.5

        # 定义qkv全连接层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 注意力Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        # 输出全连接层
        self.proj = nn.Linear(dim, dim)
        # 输出Dropout
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # 计算qkv并调整形状
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 计算输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 稀疏注意力块类
class SABlock(nn.Module):
    def __init__(self, dim, num_heads, sparse_size=0, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # 位置嵌入
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # 归一化层
        self.norm1 = norm_layer(dim)
        # 注意力机制
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # DropPath层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # MLP隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        self.sparse_size = sparse_size
        if self.ls:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            # 层缩放参数
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x_befor = x.flatten(2).transpose(1, 2)
        B, N, H, W = x.shape

        # 稀疏化处理
        x, Ho, Hp, C = alter_sparse(x, self.sparse_size)

        Bf, Nf, Hf, Wf = x.shape
        x = x.flatten(2).transpose(1, 2)
        # 注意力计算
        x = self.attn(self.norm1(x))
        x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)

        # 去稀疏化处理
        x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
        x = x.flatten(2).transpose(1, 2)

        # 是否使用层缩放（layer scale）
        if self.ls:
            """
               1. **使用层缩放（`self.ls` 为 `True`）**：
                   - 在计算完注意力和 MLP 之后，结果乘以可学习的缩放参数 `gamma_1` 和 `gamma_2`。
                   - 这种缩放可以控制每一层对最终输出的影响，通常用于改善训练稳定性。
            """
            x = x_befor + self.drop_path(self.gamma_1 * x)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))
        else:
            """
               2. **不使用层缩放（`self.ls` 为 `False`）**：
                   - 没有乘以缩放参数，直接将注意力和 MLP 的输出通过残差连接。
                   - 这种方式更简单，但可能在某些情况下不如使用缩放的方式稳定。
            """
            x = x_befor + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # 调整输出形状
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x

if __name__ == '__main__':
    # 创建 SABlock 实例
    sa_block = SABlock(dim=64, num_heads=8, sparse_size=4)

    # 创建一个输入张量 (batch_size, channels, height, width)
    x = torch.randn(1, 64, 8, 8)

    # 调用 SABlock 的 forward 方法
    output = sa_block(x)
    # 输出结果的形状
    print("Output shape:", output.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")