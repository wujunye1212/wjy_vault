import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import DropPath, trunc_normal_
from typing import List
from typing import Tuple
import sys
import os

# 用于计算二维相对位置编码
class RetNetRelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2)) # 用于计算相对位置的正弦和余弦编码. 通过torch.linspace生成线性递增的序列，再利用特定公式计算出角度参数。
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)) # 衰减因子,用于表示空间距离对注意力权重的影响，随着距离的增加,衰减因子会降低. 这个衰减因子与注意力机制结合使用,用于模拟随着空间距离增加注意力逐渐减少的现象。
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    # 根据不同的空间位置，计算每个像素与其他像素之间的曼哈顿距离，并用之前计算的 decay 参数来调整衰减程度
    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay) # 高度的索引
        index_w = torch.arange(W).to(self.decay) # 宽度的索引
        grid = torch.meshgrid([index_h, index_w]) # 生成索引网格,它创建了所有可能的 (h, w) 组合。
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2) # 将网格堆叠起来,并通过 reshape 将其转换为(H*W, 2)的形状,这表示每个像素的二维坐标
        mask = grid[:, None, :] - grid[None, :, :]  # 计算每个像素与其他所有像素之间的曼哈顿距离(通过减法和取绝对值完成),得到一个大小为(H*W, H*W, 2)的矩阵
        mask = (mask.abs()).sum(dim=-1) # 对最后一个维度求和，得到曼哈顿距离
        mask = mask * self.decay[:, None, None]  # 将计算出的距离与衰减因子相乘，得到最终的二维衰减掩码: (n,H*W,H*W),n是注意力头的数量
        return mask

    # 生成的是一维的衰减掩码，主要用于沿一个维度的注意力计算
    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay) # 一维索引
        mask = index[:, None] - index[None, :]  # 计算一维索引之间的绝对差,得到曼哈顿距离,shape是(l,l)
        mask = mask.abs()  # (l,l)
        mask = mask * self.decay[:, None, None]  # 将距离与衰减因子相乘,得到一维衰减掩码: (n,l,l), 为每一个注意力头都生成一个不同衰减掩码矩阵
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1)) # 计算正弦位置编码.  slen[0]和 slen[1]分别代表输入特征图的高度和宽度,二者相乘表示特征图的总像素数
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1)) # 计算余弦位置编码
            retention_rel_pos = ((sin, cos), self.decay.exp()) # 与衰减因子结合,形成位置编码对

        elif chunkwise_recurrent:
            index = torch.arange(slen[0] * slen[1]).to(self.decay) # 生成一个从0到 slen[0]*slen[1]-1 的一维索引向量（即图像的像素位置索引）
            sin = torch.sin(index[:, None] * self.angle[None, :])  # 计算正弦位置编码,index[:, None]将索引扩展为(l, 1)的shape; angle[None, :]扩展为 (1, d1)的shape), 所以输出的shape是: (l,d1)
            sin = sin.reshape(slen[0], slen[1], -1)  # 将正弦编码重塑为三维张量 (h,w,d1),其中h和w是特征图的高度和宽度
            cos = torch.cos(index[:, None] * self.angle[None, :])  # 计算余弦位置编码，与正弦编码计算类似,输出的shape是:(l,d1)
            cos = cos.reshape(slen[0], slen[1], -1)  # 将余弦编码重塑为(h,w,d1)的形状

            mask_h = self.generate_1d_decay(slen[0]) # 生成H方向的1D衰减掩码，大小为 (num_heads,h,h), 用于注意力计算中的权重调整
            mask_w = self.generate_1d_decay(slen[1]) # 生成W方向的1D衰减掩码，大小为 (num_heads,w,w), 用于注意力计算中的权重调整

            retention_rel_pos = ((sin, cos), (mask_h, mask_w)) # 将正弦和余弦位置编码与生成的h和w方向的衰减掩码组合在一起

        else:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])  # (l,d1)
            sin = sin.reshape(slen[0], slen[1], -1)  # (h,w,d1)
            cos = torch.cos(index[:, None] * self.angle[None, :])  # (l,d1)
            cos = cos.reshape(slen[0], slen[1], -1)  # (h,w,d1)
            mask = self.generate_2d_decay(slen[0], slen[1])  # (n,l,l)  生成二维的衰减掩码,大小为 (n,l,l),这里l是特征图的总像素数(即h*w). 这个掩码考虑了所有像素之间的距离，用于全局的注意力计算。
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

# 将输入张量的奇数和偶数维度分别取出，并进行旋转操作，这是为了与位置编码的正弦和余弦部分进行匹配。
def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2] # (b,n,h,w,d1/2)
    x2 = x[:, :, :, :, 1::2] # (b,n,h,w,d1/2)
    x = torch.stack([-x2, x1], dim=-1) # (b,n,h,w,d1/2,2)
    out = x.flatten(-2) # (b,n,h,w,d1)
    return out

# 将输入张量 x 与 sin 和 cos 进行组合，从而实现对特征的相对位置调制。
def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)  # x:(b,n,h,w,d1)  sin:(h,w,d1)  cos:(h,w,d1)


class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2) #(b c h w)
        x = self.conv(x) #(b c h w)
        x = x.permute(0, 2, 3, 1) #(b h w c)
        return x


class VisionRetentionChunk(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos # sin, cos:(h,w,d1)  mask_h:(n,h,h)  mask_w:(n,w,w)

        q = self.q_proj(x) # (b,h,w,c)-->(b,h,w,c)
        k = self.k_proj(x) # (b,h,w,c)-->(b,h,w,c)
        v = self.v_proj(x) # (b,h,w,c)-->(b,h,w,c)
        lepe = self.lepe(v) # (b,h,w,c)-->(b,h,w,c)

        k *= self.scaling # 相当于除以根号dk
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b,h,w,c)-->(b,h,w,n,d1)-->(b,n,h,w,d1)   c=n*d1,n是注意力头的数量,d1是每个头的通道数
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b,h,w,c)-->(b,h,w,n,d1)-->(b,n,h,w,d1)
        qr = theta_shift(q, sin, cos) # q:(b,n,h,w,d1)  sin:(h,w,d1)  cos:(h,w,d1)  qr:(b,n,h,w,d1)
        kr = theta_shift(k, sin, cos) # k:(b,n,h,w,d1)  sin:(h,w,d1)  cos:(h,w,d1)  kr:(b,n,h,w,d1)

        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''

        qr_w = qr.transpose(1, 2)  # (b,n,h,w,d1)-->(b,h,n,w,d1)
        kr_w = kr.transpose(1, 2)  # (b,n,h,w,d1)-->(b,h,n,w,d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)  # (b,h,w,c)-reshape->(b,h,w,n,d2)-permute->(b,h,n,w,d2)  因为factor==1,所以d2==d1

        # 首先在W方向上执行注意力
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)  # (b,h,n,w,d1) @ (b,h,n,d1,w) == (b,h,n,w,w)
        qk_mat_w = qk_mat_w + mask_w  # (b,h,n,w,w) + (n,w,w) == (b,h,n,w,w)
        qk_mat_w = torch.softmax(qk_mat_w, -1)  # (b,h,n,w,w)
        v = torch.matmul(qk_mat_w, v)  # (b,h,n,w,w) @ (b,h,n,w,d2) == (b,h,n,w,d2)


        qr_h = qr.permute(0, 3, 1, 2, 4)  # (b,n,h,w,d1)-permute->(b,w,n,h,d1)
        kr_h = kr.permute(0, 3, 1, 2, 4)  # (b,n,h,w,d1)-permute->(b,w,n,h,d1)
        v = v.permute(0, 3, 2, 1, 4)  # (b,h,n,w,d2)-permute->(b,w,n,h,d2)

        # 然后在H方向上执行注意力
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)  # (b,w,n,h,d1) @ (b,w,n,d1,h) == (b,w,n,h,h)
        qk_mat_h = qk_mat_h + mask_h  # (b,w,n,h,h) + (n,h,h) == (b,w,n,h,h)
        qk_mat_h = torch.softmax(qk_mat_h, -1)  # (b,w,n,h,h)
        output = torch.matmul(qk_mat_h, v)  # (b,w,n,h,h) @ (b,w,n,h,d2) == (b,w,n,h,d2)

        # 通过引入DWConv来增强局部表达能力,即lepe
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # (b,w,n,h,d2)-permute->(b,h,w,n,d2)-flatten->(b,h,w,n*d2)
        output = output + lepe # (b,h,w,n*d2) + (b,h,w,c) = (b,h,w,c),  c=n*d1, d1==d2
        output = self.out_proj(output) # (b,h,w,c)-->(b,h,w,c)
        return output



if __name__ == '__main__':
    # (B,H,W,C)   通道C要能被注意力头的2倍整除,两种参数设置： (1) C==64,num_head==4;  (2) C==96,num_head==3
    inputs = torch.randn(1,224,224,96)
    b, h, w, c = inputs.size()

    # 定义模型
    Model = VisionRetentionChunk(embed_dim=96, num_heads=3)
    # 用于计算二维相对位置编码
    Relpos = RetNetRelPos2d(embed_dim=96, num_heads=3, initial_value=1, heads_range=3)

    # 计算token相对位置
    rel_pos = Relpos((h, w), chunkwise_recurrent=True)
    out = Model(inputs,rel_pos)
    print(out.shape)