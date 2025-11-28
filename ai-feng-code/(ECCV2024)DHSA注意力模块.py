import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
Conv2d = nn.Conv2d
'''
ECCV 2024
即插即用注意力模块：DHSA 动态范围直方图自注意力模块
用于应对恶劣天气条件下的图像恢复任务（如去除雨滴、雪、雾等），
本文的核心创新是动态范围直方图自注意力 (Dynamic-range Histogram Self-Attention, DHSA) 模块，
能够更好地处理恶劣天气下的图像，捕捉全局和局部的动态特征。

DHSA模块原理及作用：

动态范围卷积：该模块首先通过对输入特征进行水平和垂直排序，重新排列特征的空间分布，
将高强度和低强度的像素重新排列到矩阵的对角线上。
这样，卷积可以在动态范围内进行计算，从而允许卷积核更有效地处理与天气相关的劣化特征。

直方图自注意力：DHSA根据像素强度将空间特征划分为多个直方图bin，并在bin内和bin之间执行自注意力操作。
这种机制允许模型针对恶劣天气下的动态范围特征进行更精确的聚焦和处理，从而有效区分天气劣化区域和背景区域。

双分支设计：DHSA包含两种重塑方式，即bin-wise直方图重塑 (BHR) 和frequency-wise直方图重塑 (FHR)，
分别提取全局和局部的信息。这种设计通过双路径注意力机制，进一步增强了全局和局部特征的提取能力，
有助于在恢复过程中处理复杂的天气条件。

作用：
DHSA模块能够更好地建模空间上动态的、天气引起的劣化特征，尤其是在长距离特征聚合方面表现优越。
通过将类似强度的像素聚类，并针对这些聚类执行自注意力操作，
DHSA能够更加高效地捕捉恶劣天气下的图像退化模式，从而提升图像恢复的效果。

适用于：图像恢复，图像去噪、雨、雪、雾，目标检测，图像增强等所有CV2维任务通用
'''
## Dynamic-range Histogram Self-Attention (DHSA)
class DHSA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, ifBox=True):
        super(DHSA, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)  # * self.weight + self.bias

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        x_sort, idx_h = x[:, :c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:, :c // 2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # b,c,x,x

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:, :c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c // 2] = out_replace
        return out

# 输入 B C H W,  输出B C H W
if __name__ == "__main__":
    # 创建DHSA模块的实例
    model = DHSA(64)
    input = torch.randn(1, 64, 128, 128)
    # 执行前向传播
    output= model(input)
    print('Input size:', input.size())
    print('Output size:', output.size())
