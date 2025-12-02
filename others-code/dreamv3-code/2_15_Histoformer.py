import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 一图看懂torch.gather()函数用法: https://zhuanlan.zhihu.com/p/661293803

Conv2d = nn.Conv2d

class Attention_histogram(nn.Module):
    def __init__(self, dim, num_heads, bias, ifBox=True):
        super(Attention_histogram, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1] # x:(B,C,HW)
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw] # [0,0]
        x = F.pad(x, t_pad, 'constant', 0) # (B,C,HW)-pad->(B,C,HW)
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
        q, t_pad = self.pad(q, self.factor) # (B,C,HW)-pad->(B,C,HW)
        k, t_pad = self.pad(k, self.factor) # (B,C,HW)-pad->(B,C,HW)
        v, t_pad = self.pad(v, self.factor) # (B,C,HW)-pad->(B,C,HW)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads) # (B,C,HW)-rearrange->(B,k,dk,hw)   C=k*d, HW=k*hw  与书中对应, 不考虑这里的batchsize(B), 余下的k/dk/hw分别对应: dk与书中的C对应, k与书中的B对应, hw与书中的HW/B对应
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads) # (B,C,HW)-rearrange->(B,k,dk,hw)   C=k*d, HW=k*hw
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads) # (B,C,HW)-rearrange->(B,k,dk,hw)   C=k*d, HW=k*hw
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature # (B,k,dk,hw) @ (B,k,hw,dk) = (B,k,dk,dk)
        attn = self.softmax_1(attn, dim=-1) # (B,k,dk,dk)-->(B,k,dk,dk)
        out = (attn @ v) # (B,k,dk,dk) @ (B,k,dk,hw) = (B,k,dk,hw)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)  # (B,k,dk,hw) -rearrange-> (B,C,HW)
        out = self.unpad(out, t_pad) # (B,C,HW)-unpad->(B,C,HW)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        x_sort, idx_h = x[:, :c // 2].sort(-2) # (B,C,H,W)-->(B,C/2,H,W)  在H维度进行排序,x_sort是排序后的张量,idx_h是排序后的索引
        x_sort, idx_w = x_sort.sort(-1) # 在W维度进行排序,排序操作在已经按H维度排序后的张量x_sort上进行。 idx_w是按照W维度排序后的索引
        x[:, :c // 2] = x_sort # 将排序后的张量 x_sort 赋值回原始张量x的前c//2个通道
        qkv = self.qkv_dwconv(self.qkv(x)) # (B,C,H,W)-qkv->(B,5C,H,W)-dwconv->(B,5C,H,W)
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # 分割为Value特征和两对QK特征,每一个的shape都是(B,C,H,W)

        v, idx = v.view(b, c, -1).sort(dim=-1) # 在V矩阵的HW方向进行排序,并记录索引: (B,C,H,W)-view->(B,C,HW)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx) # (B,C,H,W)-view->(B,C,HW)  按照v矩阵的索引, 来重新排列q1
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx) # (B,C,H,W)-view->(B,C,HW)  按照v矩阵的索引, 来重新排列k1
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx) # (B,C,H,W)-view->(B,C,HW)  按照v矩阵的索引, 来重新排列q2
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx) # (B,C,H,W)-view->(B,C,HW)  按照v矩阵的索引, 来重新排列k2

        out1 = self.reshape_attn(q1, k1, v, True) # 执行BHR注意力: (B,C,HW)-->(B,C,HW)
        out2 = self.reshape_attn(q2, k2, v, False) # 执行FHR注意力:(B,C,HW)-->(B,C,HW)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w) #将排序后的元素按照之前的索引,恢复到之前的位置(B,C,HW)-scatter->(B,C,HW)--view->(B,C,H,W)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w) #将排序后的元素按照之前的索引,恢复到之前的位置(B,C,HW)-scatter->(B,C,HW)--view->(B,C,H,W)
        out = out1 * out2 # 将两个输出执行逐元素乘法进行融合: (B,C,H,W) * (B,C,H,W) = (B,C,H,W)
        out = self.project_out(out) # 执行1×1Conv进行变换: (B,C,H,W)-->(B,C,H,W)

        out_replace = out[:, :c // 2] # 取前C/2个通道,恢复最初的顺序: (B,C/2,H,W)
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace) # 将输出按照最早W方向排序的索引,进行恢复: (B,C/2,H,W)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace) # 将输出按照最早H方向排序的索引,进行恢复: (B,C/2,H,W)
        out[:, :c // 2] = out_replace # (B,C,H,W)
        return out


if __name__ == '__main__':
    # (B,C,H,W)
    x1 = torch.randn(1, 64, 224,224)

    Model = Attention_histogram(dim=64, num_heads=8, bias=False, ifBox=True)
    out = Model(x1)
    print(out.shape)