import math
import torch
import torch.nn as nn
from einops import rearrange
import warnings
warnings.filterwarnings('ignore')
# 论文：https://www.sciencedirect.com/science/article/abs/pii/S0262885624002956   SCI 2024
'''
         LDConv：用于改进卷积神经网络的线性可变形卷积        SCI 2024
基于卷积运算的神经网络在深度学习领域取得了显著的成果，但标准卷积运算存在两个固有的缺陷。
一方面，卷积运算被限制在局部窗口内，因此无法从其他位置捕获信息，并且其采样形状是固定的。
另一方面，卷积核的大小固定为 k*k，例如 1*1, 3*3, 5*5 和 7*7等这是一个固定的正方形形状，
参数的数量往往随着卷积核大小的增加而趋于平方增长。

虽然Deformable Convolution（Deformable Conv）解决了标准卷积的固定采样问题，但参数数量也趋于平方增长，
Deformable Conv 只能定义 k*k 卷积运算提取特征。

针对上述问题，本文探讨了线性可变形卷积（LDConv），该算法为卷积核提供了任意数量的参数和任意采样形状，
为网络开销和性能之间的权衡提供了更丰富的选择。在LDConv中，定义了一种新的坐标生成算法，
用于为任意大小的卷积核生成不同的初始采样位置。为了适应不断变化的目标，引入了偏移来调整每个位置的样品形状。
LDConv 将标准卷积和可变形卷积的参数数量的增长趋势修正为线性增长。

与Deformable Conv相比，LDConv提供了更丰富的选择，当LDConv的参数数量设置为K的平方时，LDConv可以等效于可变形卷积。
LDConv通过不规则卷积运算完成了高效的特征提取过程，为卷积采样形状带来了更多的探索选项。
在代表性数据集COCO2017、VOC 7 + 12和VisDrone-DET2021上的目标检测实验充分证明了LDConv的优势。

LDConv是一种即插即用的卷积模块，可以替代模型中的任何卷积模块，提高各种模型性能。

适用于所有CV任务的即插即用线性可变性卷积模块
'''
class LDConv_2D(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv_2D, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the LDConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset




class LDConv_3D(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv_3D, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv3d(inc, outc, kernel_size=(num_param, 1, 1), stride=(num_param, 1, 1), bias=bias),
            nn.BatchNorm3d(outc),
            nn.SiLU())  # 这部分将添加BN和SiLU，类似YOLOv5中的原始Conv操作。
        self.p_conv = nn.Conv3d(inc, 3 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = tuple(gi * 0.1 for gi in grad_input)
        grad_output = tuple(go * 0.1 for go in grad_output)

    def forward(self, x):
        # N 是 num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 3
        # (b, 3N, d, h, w)
        p = self._get_p(offset, dtype)

        # (b, d, h, w, 3N)
        p = p.contiguous().permute(0, 2, 3, 4, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                          torch.clamp(q_lt[..., N:2 * N], 0, x.size(3) - 1),
                          torch.clamp(q_lt[..., 2 * N:], 0, x.size(4) - 1)], dim=-1).long()

        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                          torch.clamp(q_rb[..., N:2 * N], 0, x.size(3) - 1),
                          torch.clamp(q_rb[..., 2 * N:], 0, x.size(4) - 1)], dim=-1).long()

        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:2 * N], q_lt[..., 2 * N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:2 * N], q_rb[..., 2 * N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1),
                       torch.clamp(p[..., N:2 * N], 0, x.size(3) - 1),
                       torch.clamp(p[..., 2 * N:], 0, x.size(4) - 1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * \
               (1 + (q_lt[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * \
               (1 + (q_lt[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * \
               (1 - (q_rb[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * \
               (1 - (q_rb[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * \
               (1 - (q_lb[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * \
               (1 + (q_lb[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * \
               (1 + (q_rt[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * \
               (1 - (q_rt[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)
        return out
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        depth_number = round(self.num_param ** (1 / 3))
        row_number = self.num_param // (base_int * depth_number)
        mod_number = self.num_param % (base_int * depth_number)

        p_n_z, p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, depth_number),
            torch.arange(0, row_number),
            torch.arange(0, base_int)
        )
        p_n_z = torch.flatten(p_n_z)
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)

        if mod_number > 0:
            mod_p_n_z, mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(depth_number, depth_number + 1),
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number)
            )
            mod_p_n_z = torch.flatten(mod_p_n_z)
            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_z, p_n_x, p_n_y = torch.cat((p_n_z, mod_p_n_z)), torch.cat((p_n_x, mod_p_n_x)), torch.cat(
                (p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_z, p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 3 * N, 1, 1, 1).type(dtype)
        return p_n
    def _get_p_0(self, d, h, w, N, dtype):
        p_0_z, p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, d * self.stride, self.stride),
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride)
        )
        p_0_z = torch.flatten(p_0_z).view(1, 1, d, h, w).repeat(1, N, 1, 1, 1)
        p_0_x = torch.flatten(p_0_x).view(1, 1, d, h, w).repeat(1, N, 1, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, d, h, w).repeat(1, N, 1, 1, 1)
        p_0 = torch.cat([p_0_z, p_0_x, p_0_y], 1).type(dtype)
        return p_0
    def _get_p(self, offset, dtype):
        N, d, h, w = offset.size(1) // 3, offset.size(2), offset.size(3), offset.size(4)

        # (1, 3N, 1, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 3N, d, h, w)
        p_0 = self._get_p_0(d, h, w, N, dtype)
        p = p_0 + p_n + offset
        return p
    def _get_x_q(self, x, q, N):
        b, d, h, w, _ = q.size()
        padded_w = x.size(4)
        padded_h = x.size(3)
        c = x.size(1)
        # (b, c, d*h*w)
        x = x.contiguous().view(b, c, -1)
        # (b, d, h, w, N)
        index = q[..., :N] * (padded_h * padded_w) + q[..., N:2 * N] * padded_w + q[...,2 * N:]  # offset_z*h*w + offset_x*w + offset_y
        # (b, c, d*h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, d, h, w, N)
        return x_offset
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, d, h, w, n = x_offset.size()

        x_offset = rearrange(x_offset, 'b c d h w n -> b c (d n) h w')
        return x_offset

if __name__ == '__main__':
    # input = torch.rand(1, 32, 256, 256) #输入 B C H W,
    input = torch.rand(1,32,16,256,256) #输入 B C D  H W,

    #LDConv_2D   # 输入 B C H W,  输出 B C H W
    # model = LDConv_2D(inc=32,outc=32,num_param=3)

    #LDConv_3D   # 输入B C D H W,  输出 B C D H W
    model = LDConv_3D(inc=32,outc=32,num_param=3)
    output = model (input)
    print('input_size:', input.size())
    print('output_size:', output.size())
