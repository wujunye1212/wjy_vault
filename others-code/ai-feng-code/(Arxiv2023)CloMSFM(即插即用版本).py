import torch
import torch.nn as nn
from typing import List

'''

论文题目：重新思考轻量级视觉转换器中的局部感知
CloMSFM：多尺度特征融合模块，能够同时捕获高频和低频信息

在CloFormer中，我们引入了一个名为AttnConv的卷积算子，
它采用了注意力的风格，并充分利用了共享权重和上下文感知权重的优势来进行局部感知。
此外，它还使用了一种新的方法，该方法结合了比普通的局部自我注意更强的非线性来生成上下文感知权重。

在CloFormer 中，我们采用双分支架构，
其中一个分支使用 AttnConv 捕获高频信息，
而另一个分支使用 vanilla 注意力和下采样捕获低频信息。
双分支结构使CloFormer能够同时捕获高频和低频信息
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

class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            MemoryEfficientSwish(),
            nn.Conv2d(dim, dim, 1, 1, 0)
            # nn.Identity()
        )
    def forward(self, x):
        return self.act_block(x)
class CloMSFM(nn.Module):

    def __init__(self, dim, num_heads, group_split: List[int], kernel_sizes: List[int], window_size=7,
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        # projs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3 * self.dim_head * group_head, 3 * self.dim_head * group_head, kernel_size,
                                   1, kernel_size // 2, groups=3 * self.dim_head * group_head))
            act_blocks.append(AttnMap(self.dim_head * group_head))
            qkvs.append(nn.Conv2d(dim, 3 * group_head * self.dim_head, 1, 1, 0, bias=qkv_bias))
            # projs.append(nn.Linear(group_head*self.dim_head, group_head*self.dim_head, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1] * self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1] * self.dim_head * 2, 1, 1, 0, bias=qkv_bias)
            # self.global_proj = nn.Linear(group_split[-1]*self.dim_head, group_split[-1]*self.dim_head, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size != 1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        qkv = to_qkv(x)  # (b (3 m d) h w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()  # (3 b (m d) h w)
        q, k, v = qkv  # (b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v)  # (b (m d) h w)
        return res

    def low_fre_attention(self, x: torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        b, c, h, w = x.size()
        q = to_q(x).reshape(b, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()
        kv = avgpool(x)
        kv = to_kv(kv)

        # 确保总元素数一致
        num_elements = kv.numel()
        num_groups = num_elements // (b * 2 * self.dim_head)

        # 检查是否整除
        if (num_elements % (b * 2 * self.dim_head)) != 0:
            raise RuntimeError(
                f"Invalid shape: {num_elements} elements cannot be reshaped to ({b}, 2, -1, {self.dim_head})")

        kv = kv.view(b, 2, -1, self.dim_head, num_groups).permute(1, 0, 2, 4, 3).contiguous()
        k, v = kv
        attn = self.scalor * q @ k.transpose(-1, -2)
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res
    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        # 注释的这行是原论文作者的代码
        # return  self.proj_drop(self.proj(torch.cat(res, dim=1)))
        return x+ self.proj_drop(self.proj(torch.cat(res, dim=1)))


if __name__ == '__main__':
    # 实例化模型
    # dim = 64      #表示输入特征图的通道数
    # num_heads = 8 #表示多头注意力中的头数
    # group_split = [8, 8] #定义了不同的组
    # kernel_sizes = [3]  #定义了不同的组相应的卷积核大小
    # window_size = 7    # 低频注意力的窗口大小。
    #CloMSFM多尺度特点融合模块：能够同时捕获高频和低频信息
    # model = CloMSFM(dim=8, num_heads=8, group_split=[4, 4], kernel_sizes=[3], window_size=7)
    # # 输入一张随机图片
    # input = torch.randn(1, 8,56,56)
    # # 前向传播
    # output = model(input)   #  输入 B C H W,  输出 B C H W
    # print('input_size:',input.size())
    # print('output_size:',output.size())
    # 示例：初始化模型和执行前向传播
    dim = 256     #表示输入特征图的通道数
    num_heads = 8   #表示多头注意力中的头数
    #第一种分组，这个group_split，kernel_sizes自己可以自行修改哈，
    # 注意：group_split 长度比kernel_sizes 大1 。
    # group_split = [2, 2, 2, 2]  #定义了不同的组  长度是4
    # kernel_sizes = [3, 5, 7]   #定义了不同的组相应的卷积核大小 长度是3
    #第二种分组
    group_split = [4,4]  #定义了不同的组
    kernel_sizes = [3]   #定义了不同的组相应的卷积核大小


    window_size = 7       # 低频注意力的窗口大小。
    # CloMSFM多尺度特点融合模块：能够同时捕获高频和低频信息
    model = CloMSFM(dim, num_heads, group_split, kernel_sizes, window_size)
    # 输入：B C H W  --> B C H W
    input = torch.randn(1, dim, 64, 64)
    output = model(input)
    print('input_size:',input.size())
    print('output_size:',output.size())


