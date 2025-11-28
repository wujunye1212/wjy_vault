import torch
import torch.nn as nn
from typing import Tuple, Union

from ultralytics.nn.modules import C3

'''
来自 CVPR 2024 顶会论文
即插即用模块：MaSA曼哈顿自注意力  也是一个对MHSA多头自注意力模块的改进

近年来，Vision Transformer （ViT） 在计算机视觉领域越来越受到关注。
然而，ViT 的核心组件 Self-Attention 缺乏明确的空间先验，并且具有二次计算复杂性，从而限制了 ViT 的适用性。
为了缓解这些问题，我们从 NLP 领域最近的 Retentive Network （RetNet） 中汲取灵感，
并提出了 RMT，这是一种具有明确空间先验的强大视觉骨干。

具体来说，我们将 RetNet 的时间衰减机制扩展到空间域，并提出了一个基于曼哈顿距离的空间衰减矩阵，
以引入自我注意之前的显式空间。此外，本文还提出了一种善于适应显式空间先验的注意力分解形式，
旨在不破坏空间衰减矩阵的情况下减轻对全局信息进行建模的计算负担。基于空间衰减矩阵和注意力分解形式，
我们可以灵活地将显式空间先验集成到线性复杂度的视觉主干中。

大量实验表明，在各种视觉任务中表现出卓越的性能。具体来说，在没有额外训练数据的情况下，
RMT 在 ImageNet-1k 上以 27M/4.5GFLOPs 和 96M/18.2GFLOPs 实现了 84.8% 和 86.1% 的 top-1 acc。
对于下游任务，RMT 在 COCO 检测任务上实现了 54.5 box AP 和实例分割 47.2 mask AP，
在 ADE20K 语义分割任务上实现了 52.8 mIoU。

适用于：图像分类，目标检测、实例分割和语义分割等所有计算机视觉CV任务通用的即插即用模块
'''

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)  # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x
class RetNetRelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads=4, initial_value=1, heads_range=3):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # (H*W 2)
        mask = grid[:, None, :] - grid[None, :, :]  # (H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]  # (n H*W H*W)
        return mask

    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        mask = mask * self.decay[:, None, None]  # (n l l)
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=True):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())

        elif chunkwise_recurrent:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])  # (l d1)
            sin = sin.reshape(slen[0], slen[1], -1)  # (h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :])  # (l d1)
            cos = cos.reshape(slen[0], slen[1], -1)  # (h w d1)

            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])  # (l d1)
            sin = sin.reshape(slen[0], slen[1], -1)  # (h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :])  # (l d1)
            cos = cos.reshape(slen[0], slen[1], -1)  # (h w d1)
            mask = self.generate_2d_decay(slen[0], slen[1])  # (n l l)
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

class MaSA(nn.Module):

    def __init__(self, embed_dim, num_heads=4, value_factor=1):
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
        self.RetNetRelPos2d = RetNetRelPos2d(embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        x = x.permute(0,2,3,1)
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = self.RetNetRelPos2d((h,w))

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)
        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''
        qr_w = qr.transpose(1, 2)  # (b h n w d1)
        kr_w = kr.transpose(1, 2)  # (b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)  # (b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)  # (b h n w w)
        qk_mat_w = qk_mat_w + mask_w  # (b h n w w)
        qk_mat_w = torch.softmax(qk_mat_w, -1)  # (b h n w w)
        v = torch.matmul(qk_mat_w, v)  # (b h n w d2)

        qr_h = qr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        v = v.permute(0, 3, 2, 1, 4)  # (b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)  # (b w n h h)
        qk_mat_h = qk_mat_h + mask_h  # (b w n h h)
        qk_mat_h = torch.softmax(qk_mat_h, -1)  # (b w n h h)
        output = torch.matmul(qk_mat_h, v)  # (b w n h d2)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # (b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)

        return output.permute(0,3,1,2)

# 输入 B H W C , 输出 B  H W C
if __name__ == "__main__":
    module = MaSA(64)  # 创建 MaSA模块实例，输入通道数为 64
    input_tensor = torch.randn(1,64,128, 128)  # 创建一个形状为 (1, 64,128, 128) 的随机输入张量
    output_tensor = module(input_tensor)  # 通过MaSA模块计算输出
    print('Input size:', input_tensor.size())  # 打印输入张量的形状
    print('Output size:', output_tensor.size())  # 打印输出张量的形状