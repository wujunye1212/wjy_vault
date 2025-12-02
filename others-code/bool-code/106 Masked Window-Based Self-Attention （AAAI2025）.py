from einops import rearrange
import numbers
import torch
from torch import einsum
import torch.nn as nn

"""
    论文地址：https://arxiv.org/abs/2404.07846
    论文题目：Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising（AAAI 2025）
    中文题目：基于Transformer的自监督图像去噪盲点网络的再思考(AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV1MBNweWENL/ 
        掩码窗口自注意力机制（Masked Window-Based Self-Attention ,M-WSA）：
            理论支撑：旨在模仿扩张卷积行为，满足盲点网络（BSN）的要求，同时增强局部拟合能力和扩大感受野。
            实现方式：在传统的窗口注意力中，每个查询Q会与其窗口内的所有空间位置上的键K/值V相互作用。在M-WSA中，仅关注窗口内偶数坐标的位置。【个人认为可以加一个，轻量化】
            思想延伸（跨领域写作）：【二次创新与写作思路 请务必看视频】
"""


# 将输入张量从 BCWH 转换为 B(HW)C
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

# 将输入张量从 B(HW)C 转换为 BCWH
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

# 无偏置的层归一化
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

# 带偏置的层归一化
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# 在指定维度扩展张量
def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

# 计算一维相对位置logits
def relative_logits_1d(q, rel_k):
    """
    计算一维相对位置logits。

    Args:
        q: 查询张量，形状为 (batch_size, height, width, dim_head)。
        rel_k: 相对位置键张量，形状为 (2 * rel_size - 1, dim_head)。

    Returns:
        相对位置logits，形状为 (batch_size, height, width, rel_size)。
    """
    b, h, w, _ = q.shape  # 获取查询张量的形状
    r = (rel_k.shape[0] + 1) // 2  # 计算相对大小 rel_size

    # 使用爱因斯坦求和约定计算 logits
    # b: batch_size, x: height, y: width, d: dim_head, r: rel_size
    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')  # 重新排列 logits，方便后续计算
    logits = rel_to_abs(logits)  # 将相对位置索引转换为绝对位置索引

    logits = logits.reshape(b, h, w, r)  # 恢复 logits 的形状
    logits = expand_dim(logits, dim = 2, k = r) # 在维度2扩展r个维度, 假设expand_dim的功能是复制dim维度k次
    return logits

# 获取张量的设备和数据类型
def to(x):
    """
    获取张量的设备和数据类型。

    Args:
        x: 输入张量。

    Returns:
        一个字典，包含张量的设备和数据类型。
    """
    return {'device': x.device, 'dtype': x.dtype}

# 将相对位置索引转换为绝对位置索引
def rel_to_abs(x):
    """
    将相对位置索引转换为绝对位置索引。

    Args:
        x: 输入张量，形状为 (batch_size, len_q, 2 * rel_size - 1)。

    Returns:
        绝对位置索引，形状为 (batch_size, len_q, rel_size)。
    """
    b, l, m = x.shape  # 获取输入张量的形状：批次大小、查询长度、相对位置编码长度
    r = (m + 1) // 2  # 计算相对大小 rel_size

    col_pad = torch.zeros((b, l, 1), **to(x))  # 创建一个填充张量，形状为 (b, l, 1)，与 x 的设备和数据类型相同
    x = torch.cat((x, col_pad), dim=2)  # 在最后一维（相对位置编码维度）拼接填充张量，形状变为 (b, l, 2 * rel_size)
    flat_x = rearrange(x, 'b l c -> b (l c)')  # 将张量展平，形状变为 (b, l * 2 * rel_size)
    flat_pad = torch.zeros((b, m - l), **to(x))  # 创建另一个填充张量，形状为 (b, 2 * rel_size - 1 - l)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)  # 在第二维（展平后的维度）拼接填充张量，形状变为 (b, l * 2 * rel_size + 2 * rel_size - 1 - l)
    final_x = flat_x_padded.reshape(b, l + 1, m)  # 重新调整张量形状为 (b, l + 1, 2 * rel_size - 1)
    final_x = final_x[:, :l, -r:]  # 获取最终的绝对位置索引，形状为 (b, l, rel_size)  取右上角的rel_size x len_q 大小的块
    return final_x

# 固定位置嵌入
class FixedPosEmb(nn.Module):
    """
    固定位置嵌入模块。用于生成固定大小窗口的注意力掩码。

    Args:
        window_size: 窗口大小。
        overlap_window_size: 重叠窗口大小。
    """
    def __init__(self, window_size, overlap_window_size):
        super().__init__()
        self.window_size = window_size  # 窗口大小
        self.overlap_window_size = overlap_window_size  # 重叠窗口大小

        """
          掩码是一个二值矩阵，根据查询（位于）和键 / 值（位于）令牌的相对位置进行掩码操作。
          当和在两个轴上的距离均为偶数时，为0，注意力值不变；否则，负无穷，经过 SoftMax 操作后注意力值变为 0，即被掩码掉。
          为提高效率，可由一个较小尺寸的二进制矩阵根据和的相对位置计算得到。
        """
        # 创建注意力掩码表，将偶数位置设置为负无穷，表示这些位置被mask
        attention_mask_table = torch.zeros((window_size + overlap_window_size - 1), (window_size + overlap_window_size - 1)) # 创建一个全零张量，大小为 (window_size + overlap_window_size - 1) x (window_size + overlap_window_size - 1)
        attention_mask_table[0::2, :] = float('-inf')  # 将偶数行设置为负无穷
        attention_mask_table[:, 0::2] = float('-inf')  # 将偶数列设置为负无穷
        attention_mask_table = attention_mask_table.view((window_size + overlap_window_size - 1) * (window_size + overlap_window_size - 1))  # 将张量展平

        # 计算窗口内每个token的成对相对位置索引
        coords_h = torch.arange(self.window_size)  # 生成高度坐标 [0, 1, 2, ..., window_size - 1]
        coords_w = torch.arange(self.window_size)  # 生成宽度坐标 [0, 1, 2, ..., window_size - 1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 生成坐标网格，形状为 (2, window_size, window_size)
        coords_flatten_1 = torch.flatten(coords, 1)  # 将坐标展平，形状为 (2, window_size * window_size)

        coords_h = torch.arange(self.overlap_window_size)  # 生成重叠窗口的高度坐标
        coords_w = torch.arange(self.overlap_window_size)  # 生成重叠窗口的宽度坐标
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 生成重叠窗口的坐标网格
        coords_flatten_2 = torch.flatten(coords, 1)  # 将重叠窗口坐标展平

        relative_coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]  # 计算相对坐标，形状为 (2, window_size * window_size, overlap_window_size * overlap_window_size)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # 调整相对坐标的维度顺序，形状为 (window_size * window_size, overlap_window_size * overlap_window_size, 2)
        relative_coords[:, :, 0] += self.overlap_window_size - 1  # 将高度坐标偏移
        relative_coords[:, :, 1] += self.overlap_window_size - 1  # 将宽度坐标偏移
        relative_coords[:, :, 0] *= self.window_size + self.overlap_window_size - 1  # 缩放高度坐标
        relative_position_index = relative_coords.sum(-1)  # 计算相对位置索引，形状为 (window_size * window_size, overlap_window_size * overlap_window_size)

        # 根据相对位置索引从掩码表中获取注意力掩码
        self.attention_mask = nn.Parameter(attention_mask_table[relative_position_index.view(-1)].view(
            1, self.window_size ** 2, self.overlap_window_size ** 2
        ), requires_grad=False)  # 设置为不可训练参数

    def forward(self):
        """
        前向传播函数。

        Returns:
            注意力掩码。
        """
        return self.attention_mask


# 相对位置嵌入
class RelPosEmb(nn.Module):
    """
    相对位置嵌入模块。

    Args:
        block_size: 块大小。
        rel_size: 相对大小。
        dim_head: 注意力头的维度。
    """
    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        height = width = rel_size  # 设置高度和宽度等于相对大小
        scale = dim_head ** -0.5  # 计算缩放因子

        self.block_size = block_size  # 块大小
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)  # 初始化高度相对位置嵌入
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)  # 初始化宽度相对位置嵌入

    def forward(self, q):
        """
        前向传播函数。

        Args:
            q: 查询张量，形状为 (batch_size, block_size * block_size, dim_head)。

        Returns:
            相对位置logits，形状为 (batch_size, block_size * block_size, block_size * block_size)。
        """
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x=block)  # 重新排列查询张量，形状变为 (b, block_size, block_size, dim_head)
        rel_logits_w = relative_logits_1d(q, self.rel_width)  # 计算宽度方向的相对位置logits
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')  # 重新排列宽度方向的相对位置logits，形状变为 (b, block_size * block_size, block_size * block_size)

        q = rearrange(q, 'b x y d -> b y x d')  # 重新排列查询张量，为高度方向计算做准备，形状变为 (b, block_size, block_size, dim_head)
        rel_logits_h = relative_logits_1d(q, self.rel_height)  # 计算高度方向的相对位置logits
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')  # 重新排列高度方向的相对位置logits，形状变为 (b, block_size * block_size, block_size * block_size)
        return rel_logits_w + rel_logits_h  # 返回宽度和高度方向的相对位置logits之和

class DilatedOCA(nn.Module):
    """
    带空洞的掩码窗口自注意力机制。

    Args:
        dim (int): 输入通道数。
        window_size (int): 窗口大小。默认为 8。
        overlap_ratio (float): 重叠比例。默认为 0.5。
        num_heads (int): 注意力头的数量。默认为 2。
        dim_head (int): 每个注意力头的维度。默认为 16。
        bias (bool): 是否使用偏置。默认为 False。
    """
    def __init__(self, dim, window_size=8, overlap_ratio=0.5, num_heads=2, dim_head=16, bias=False):
        super(DilatedOCA, self).__init__()
        self.num_spatial_heads = num_heads  # 空间注意力头的数量
        self.dim = dim  # 输入通道数
        self.window_size = window_size  # 窗口大小
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size  # 重叠窗口大小，计算方式为窗口大小加上重叠部分的大小
        self.dim_head = dim_head  # 每个注意力头的维度
        self.inner_dim = self.dim_head * self.num_spatial_heads  # 所有注意力头的总维度
        self.scale = self.dim_head**-0.5  # 缩放因子，用于缩放注意力权重

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)  # 用于展开重叠窗口的层，将输入特征图转换为重叠窗口的序列
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)  # 用于生成查询、键和值的卷积层，输入通道数为 dim，输出通道数为 inner_dim * 3
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)  # 用于输出投影的卷积层，将注意力输出的特征维度转换回输入维度 dim
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,  # 块大小，即窗口大小
            rel_size = window_size + (self.overlap_win_size - window_size), # 相对位置编码的大小，等于重叠窗口大小  self.overlap_win_size
            dim_head = self.dim_head  # 每个注意力头的维度
        )  # 相对位置嵌入层
        self.fixed_pos_emb = FixedPosEmb(window_size, self.overlap_win_size)  # 固定位置嵌入层，用于生成注意力掩码
        self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')  # 层归一化层，用于对输入特征进行归一化

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (Tensor): 输入张量，形状为 (batch_size, channels, height, width)。

        Returns:
            Tensor: 输出张量，形状与输入张量相同。
        """
        x = self.norm(x)  # 对输入特征进行层归一化
        b, c, h, w = x.shape  # 获取输入张量的形状

        qkv = self.qkv(x)  # 生成查询、键和值，形状为 (b, inner_dim * 3, h, w)

        qs, ks, vs = qkv.chunk(3, dim=1)  # 将 qkv 分成查询、键和值，每个的形状为 (b, inner_dim, h, w)

        # 空间注意力【分块】
        # 重新排列查询张量Q，将每个窗口内的特征展平，形状变为 (b * num_windows, window_size * window_size, inner_dim)
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)

        ks, vs = map(lambda t: self.unfold(t), (ks, vs))  # 使用 unfold 展开键和值，使其包含重叠窗口的信息
        # 重新排列键K和值V，形状变为 (b * num_windows, overlap_win_size * overlap_win_size, inner_dim)
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks, vs))

        # 分割多头
        # 将查询、键和值分割成多个头，形状变为 (b * num_heads, window_size * window_size, dim_head) 和 (b * num_heads, overlap_win_size * overlap_win_size, dim_head)
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_spatial_heads), (qs, ks, vs))

        # 注意力计算
        qs = qs * self.scale  # 缩放查询张量
        spatial_attn = (qs @ ks.transpose(-2, -1))  # 计算注意力权重，形状为 (b * num_heads, window_size * window_size, overlap_win_size * overlap_win_size)
        spatial_attn += self.rel_pos_emb(qs)  # 添加相对位置嵌入

        spatial_attn += self.fixed_pos_emb()  # 添加固定位置嵌入（注意力掩码）

        spatial_attn = spatial_attn.softmax(dim=-1)  # 对注意力权重进行 softmax 归一化

        out = (spatial_attn @ vs)  # 计算加权平均值，形状为 (b * num_heads, window_size * window_size, dim_head)

        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_spatial_heads, h = h // self.window_size, w = w // self.window_size, p1 = self.window_size, p2 = self.window_size)  # 重新排列输出张量，将其恢复到原始的形状

        # 合并空间和通道
        out = self.project_out(out)  # 使用卷积层进行输出投影，形状变为 (b, c, h, w)

        return out

if __name__ == "__main__":
    DilatedOCA_spatial_attn = DilatedOCA(64)
    input_tensor = torch.randn(1, 64, 128, 128)
    output_tensor = DilatedOCA_spatial_attn(input_tensor)
    # 打印输入输出尺寸
    print('输入尺寸:', input_tensor.size())  # 应输出 torch.Size([1, 64, 128, 128])
    print('输出尺寸:', output_tensor.size())  # 应保持相同尺寸
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")