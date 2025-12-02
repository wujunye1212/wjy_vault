import sys
import torch
import torch.nn as nn
from einops import rearrange
# https://arxiv.org/pdf/2403.10067
'''
论文题目：用于高光谱图像去噪的混合卷积和注意力网络     IEEE 2024
卷积和注意力特征融合模块：CAFM        
背景：  
摘要—高光谱图像（HSI）去噪对于高光谱数据的有效分析和解释至关重要。
然而，很少有人探索同时对全局和局部特征进行建模来增强 HSI 去噪。
在这篇文章中，我们提出了一种混合卷积和注意力网络（HCANet），它利用了卷积神经网络（CNNs）和Transformers的优势。
为了增强全局和局部特征的建模，我们设计了一种卷积和注意力融合模块，旨在捕获长程依赖性和邻域光谱相关性。

CAFM:所提出的卷积和注意力特征融合模块。它由本地和全局分支机构组成。
在局部分支中，采用卷积和通道洗牌进行局部特征提取。
在全局分支中，注意力机制用于对长程特征依赖关系进行建模。

适用于：高光谱图像去噪，图像增强，图像分类，目标检测，图像分割，暗光增强等所有CV2D任务
'''
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
class CAFM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv

        return output
if __name__ == '__main__':
    input = torch.rand(1, 64, 32,32)

    # #将四维转三维
    # input1 = torch.rand(1, 64, 32, 32)
    # input1 = to_3d(input1)
    # print(input1.shape)
    # # 将三维转四维
    # input1 = to_4d(input1,32,32)
    # print(input1.shape)

    CAFM = CAFM(dim=64)
    output = CAFM(input)

    print('input_size:', input.size())
    print('output_size:', output.size())