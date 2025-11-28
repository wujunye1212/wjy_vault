import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import warnings
warnings.filterwarnings('ignore')

"""
    论文地址：https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf
    论文题目：Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Reﬁnement for Image Restoration（CVPR 2024）
    中文题目：适者生存：用于图像恢复的具有注意力特征优化的自适应稀疏Transformer 
    讲解视频：https://www.bilibili.com/video/BV1vL9jYLExX/
        稀疏自注意力（Adaptive Sparse Self-Attention , ASSA）：
            实际意义：①冗余计算与噪声特征：标准 Transformer会对无信息区域进行计算，还引入冗余特征。
                    ②传统注意力机制计算复杂度高：随序列长度呈平方级增长。
                    ③平衡全局与局部信息特征
            实现方式：由基于平方 ReLU 的稀疏自注意力分支（SSA）和传统密集自注意力分支（DSA）组成，并自适应地融合两者，以更好地捕捉有用的特征交互。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

# 定义线性投影模块，用于生成注意力机制中的查询 (Q)、键 (K) 和值 (V)
class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads  # 计算总的内部维度（每个头的维度 * 头的数量）
        self.heads = heads  # 多头注意力的头数
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)  # 用于生成查询向量 Q 的线性层
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)  # 用于生成键向量 K 和值向量 V 的线性层
        self.dim = dim  # 输入特征的维度
        self.inner_dim = inner_dim  # 投影后的总维度

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape  # 输入 x 的形状为 (批量大小, 序列长度, 特征维度)
        if attn_kv is not None:  # 如果提供了外部的键值对
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)  # 扩展维度后复制 B_ 次
        else:
            attn_kv = x  # 如果没有提供外部键值对，则使用输入 x 自身
        N_kv = attn_kv.size(1)  # 键值对的序列长度

        # 生成查询 Q，形状调整为 (批量大小, 头数, 序列长度, 每头的维度)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        # 生成键 K 和值 V，形状调整为 (键值对序列长度, 批量大小, 头数, 每头的维度)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]  # 提取查询向量 Q
        k, v = kv[0], kv[1]  # 提取键向量 K 和值向量 V
        return q, k, v  # 返回 Q, K, V

# 定义稀疏窗口注意力模块
class WindowAttention_sparse(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim  # 输入特征的维度
        self.win_size = win_size  # 窗口大小 (高, 宽)
        self.num_heads = num_heads  # 多头注意力的头数
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子，默认为 1 / sqrt(每头的维度)

        # 定义相对位置偏置表，用于表示窗口内每个位置的相对偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 形状为 (窗口内位置对数, 头数)

        # 计算窗口内每个 token 的相对位置索引
        coords_h = torch.arange(self.win_size[0])  # 生成窗口高度的坐标 [0, ..., win_size[0]-1]
        coords_w = torch.arange(self.win_size[1])  # 生成窗口宽度的坐标 [0, ..., win_size[1]-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 生成网格坐标 (2, win_size[0], win_size[1])
        coords_flatten = torch.flatten(coords, 1)  # 将坐标展平为 (2, win_size[0]*win_size[1])
        # 计算相对坐标，形状为 (2, win_size[0]*win_size[1], win_size[0]*win_size[1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # 调整形状为 (win_size[0]*win_size[1], win_size[0]*win_size[1], 2)
        relative_coords[:, :, 0] += self.win_size[0] - 1  # 偏移，使相对坐标从 0 开始
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1  # 将高度和宽度的相对坐标合并为一个索引
        relative_position_index = relative_coords.sum(-1)  # 计算最终的相对位置索引
        self.register_buffer("relative_position_index", relative_position_index)  # 注册为缓冲区变量（不会参与训练）
        trunc_normal_(self.relative_position_bias_table, std=.02)  # 初始化相对位置偏置表

        # 如果 token 投影方式为线性，使用 LinearProjection 模块
        if token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")  # 如果不是线性投影方式，抛出异常

        self.token_projection = token_projection  # 保存投影方式
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力的 dropout
        self.proj = nn.Linear(dim, dim)  # 最后的线性投影层
        self.proj_drop = nn.Dropout(proj_drop)  # 投影后的 dropout

        self.softmax = nn.Softmax(dim=-1)  # softmax 用于计算注意力权重
        self.relu = nn.ReLU()  # 使用 ReLU 激活函数
        self.w = nn.Parameter(torch.ones(2))  # 定义可学习参数，用于稀疏和稠密注意力的加权

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape  # 输入 x 的形状为 (批量大小, 序列长度, 特征维度)
        q, k, v = self.qkv(x, attn_kv)  # 通过线性投影生成查询 Q、键 K 和值 V
        q = q * self.scale  # 对查询向量 Q 进行缩放
        attn = (q @ k.transpose(-2, -1))  # 计算注意力分数 (QK^T)

        # 添加相对位置偏置 == B
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # 形状为 (窗口内位置对数, 头数)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # 调整形状为 (头数, 窗口内位置对数, 窗口内位置对数)
        ratio = attn.size(-1) // relative_position_bias.size(-1)  # 计算注意力矩阵和偏置矩阵的大小比例
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)  # 扩展偏置矩阵以匹配注意力矩阵的大小

        attn = attn + relative_position_bias.unsqueeze(0)  # 将相对位置偏置添加到注意力分数中

        if mask is not None:  # 如果提供了掩码
            nW = mask.shape[0]  # 获取掩码的窗口数量
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)  # 扩展掩码以匹配注意力矩阵的大小
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)  # 添加掩码
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn0 = self.softmax(attn)  # 稠密注意力
            attn1 = self.relu(attn) ** 2  # 稀疏注意力
        else:
            attn0 = self.softmax(attn)  # 稠密注意力
            attn1 = self.relu(attn) ** 2  # 稀疏注意力

        # 使用可学习参数对稠密和稀疏注意力进行加权
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))  # 权重 1
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))  # 权重 2
        attn = attn0 * w1 + attn1 * w2  # 加权求和
        attn = self.attn_drop(attn)  # 应用 dropout

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # 根据注意力权重和值 V 计算输出
        x = self.proj(x)  # 通过全连接层
        x = self.proj_drop(x)  # 应用投影后的 dropout
        return x  # 返回最终的输出

# 将特征划分为多个窗口
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape  # 输入形状为 (批量大小, 高度, 宽度, 通道数)
    if dilation_rate != 1:  # 如果使用膨胀窗口
        x = x.permute(0, 3, 1, 2)  # 调整为 (批量大小, 通道数, 高度, 宽度)
        assert type(dilation_rate) is int, 'dilation_rate should be an integer'  # 确保膨胀率是整数
        # 使用 unfold 将特征划分为窗口
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1), stride=win_size)
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # 重新调整形状为窗口
        windows = windows.permute(0, 2, 3, 1).contiguous()  # 调整为 (窗口数, 高度, 宽度, 通道数)
    else:  # 如果不使用膨胀窗口
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)  # 将特征划分为窗口
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # 调整形状
    return windows  # 返回窗口

# 将窗口重建为原始特征
def window_reverse(windows, win_size, H, W, dilation_rate=1):
    B = int(windows.shape[0] / (H * W / win_size / win_size))  # 计算批量大小
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)  # 调整形状
    if dilation_rate != 1:  # 如果使用膨胀窗口
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # 调整形状
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)  # 使用 fold 重建特征
    else:  # 如果不使用膨胀窗口
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)  # 调整形状为原始特征
    return x  # 返回重建后的特征

# 定义 ASSA 模块，核心是基于窗口的注意力机制
class ASSA(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff', att=True,
                 sparseAtt=False):
        super().__init__()
        self.att = att  # 是否使用注意力机制
        self.sparseAtt = sparseAtt  # 是否使用稀疏注意力机制
        self.dim = dim  # 输入特征维度
        self.input_resolution = input_resolution  # 输入分辨率 (高, 宽)
        self.num_heads = num_heads  # 多头注意力的头数
        self.win_size = win_size  # 窗口大小
        self.shift_size = shift_size  # 循环平移大小
        self.mlp_ratio = mlp_ratio  # MLP 中间层扩展比率
        self.token_mlp = token_mlp  # MLP 类型

        # 如果输入分辨率小于窗口大小，则调整窗口大小
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"  # 确保平移大小合法

        self.norm1 = norm_layer(dim)  # 归一化层
        self.attn = WindowAttention_sparse(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # DropPath 用于随机丢弃路径
        self.norm2 = norm_layer(dim)  # 第二个归一化层

    def forward(self, x):
        B, L, C = x.shape  # 输入形状为 (批量大小, 序列长度, 特征维度)
        H = int(math.sqrt(L))  # 计算高度
        W = int(math.sqrt(L))  # 计算宽度

        attn_mask = None  # 初始化注意力掩码
        shortcut = x  # 保存输入作为残差连接

        x = self.norm1(x)  # 归一化
        x = x.view(B, H, W, C)  # 调整形状为 (批量大小, 高度, 宽度, 通道数)

        # 循环平移
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # 划分窗口
        x_windows = window_partition(shifted_x, self.win_size)  # 划分为窗口
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # 调整形状为 (窗口数, 窗口大小, 通道数)

        # 计算窗口内的注意力
        attn_windows = self.attn(x_windows, mask=attn_mask)  # 计算窗口注意力

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # 恢复到原始形状
        # 反向平移
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)  # 调整形状为 (批量大小, 序列长度, 特征维度)
        x = shortcut + self.drop_path(x)  # 残差连接

        return x  # 返回输出

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

if __name__ == '__main__':

    H, W = 32, 32
    input1 = torch.randn(1, 64, H, W)  #
    model = ASSA(dim=64, input_resolution=(16, 16), num_heads=8)
    input = to_3d(input1)
    output = model(input)
    output = to_4d(output,H,W)
    print('input_size:', input1.size())
    print('output_size:', output.size())