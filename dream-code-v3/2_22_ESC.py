import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.nn.init import trunc_normal_
from typing import Optional


class ConvolutionalAttention(nn.Module):
    def __init__(self, pdim: int, proj_dim_in: Optional[int] = None):
        super().__init__()
        self.pdim = pdim  # 将在“注意力分支”上做卷积的通道数
        self.proj_dim_in = proj_dim_in if proj_dim_in is not None else pdim  # 用来生成动态核 DK 的输入通道数
        self.sk_size = 3 # 动态核 DK 的空间大小固定为 3×3
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.pdim, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0)
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.pdim, x.shape[1] -self.pdim], dim=1) # (B,C,H,W)--split--->x1:(B,C1,H,W), x2:(B,C2,H,W)

            # Dynamic Conv
            bs = x1.shape[0] # B
            dynamic_kernel = self.dwc_proj(x[:, :self.pdim]).reshape(-1, 1, self.sk_size, self.sk_size) # 根据输入生成动态核：(B,C1,H,W)-dwc_proj->(B,C1*3*3,1,1)-reshape->(B*C1,1,3,3)
            x1_ = rearrange(x1, 'b c h w -> 1 (b c) h w') # (B,C1,H,W)-->(1,B*C1,H,W)
            x1_ = F.conv2d(x1_, dynamic_kernel, stride=1, padding=self.sk_size//2, groups=bs * self.pdim) # 基于动态核的卷积操作: (1,B*C1,H,W)
            x1_ = rearrange(x1_, '1 (b c) h w -> b c h w', b=bs, c=self.pdim) # (1,B*C1,H,W)--rearrange->(B,C1,H,W)

            # Static LK Conv + Dynamic Conv
            x1 = F.conv2d(x1, lk_filter, stride=1, padding=lk_filter.shape[-1] // 2) + x1_  # 基于共享大核的卷积操作 + 基于动态核的卷积结果: x1:(B,C1,H,W); 权重:(C1,C1,K,K);  输出:(B,C1,H,W)

            x = torch.cat([x1, x2], dim=1) # :(B,C1,H,W)-cat-(B,C2,H,W)-->(B,C,H,W)
        else:
            dynamic_kernel = self.dwc_proj(x[:, :self.proj_dim_in]).reshape(-1, 1, self.sk_size, self.sk_size)
            x[:, :self.pdim] = F.conv2d(x[:, :self.pdim], lk_filter, stride=1, padding=lk_filter.shape[-1] // 2) + \
                               rearrange(
                                   F.conv2d(rearrange(x[:, :self.pdim], 'b c h w -> 1 (b c) h w'), dynamic_kernel, stride=1, padding=self.sk_size//2, groups=x.shape[0] * self.pdim),
                                   '1 (b c) h w -> b c h w', b=x.shape[0]
                               )
        return x

    def extra_repr(self):
        return f'pdim={self.pdim}, proj_dim_in={self.proj_dim_in}'


# 对卷积核 做几何平均增强, k:(C1,C1,K,K)
def _geo_ensemble(k):
    k_hflip = k.flip([3]) # 沿维度 3（宽度维 W）水平翻转（horizontal flip）,得到关于垂直中轴镜像的核。
    k_vflip = k.flip([2]) # 沿维度 2（高度维 H）垂直翻转（vertical flip）,得到关于水平中轴镜像的核。
    k_hvflip = k.flip([2, 3]) # 同时在高、宽两个维度翻转，等价于旋转 180°。这是水平+垂直镜像的叠加。
    k_rot90 = torch.rot90(k, -1, [2, 3]) # 在维度 [2, 3]（H、W 平面）上旋转 -1 个 90°，即顺时针 90°（正数是逆时针）
    k_rot90_hflip = k_rot90.flip([3]) # 对已经顺时针 90° 的核再做水平翻转，相当于在 D4 群（正方形的二面体对称群）的另一个元素
    k_rot90_vflip = k_rot90.flip([2]) # 对 k_rot90 再做垂直翻转，又得到 D4 群的另一个元素
    k_rot90_hvflip = k_rot90.flip([2, 3]) # 对 k_rot90 做水平+垂直翻转（等价再旋 180°），再得到一个元素
    k = (k + k_hflip + k_vflip + k_hvflip + k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8  #将原核与其 7个几何变换逐项相加再求平均
    return k # 降低方向性偏置，让大核对各方向（边缘/结构）响应更均衡、更加“各向同性”。这与论文里“共享大核 LK 提供稳定长程交互”的思路相契合。


if __name__ == '__main__':
    # (B,C,H,W)
    x1 = torch.randn(1, 64, 224, 224)
    B, C, H, W = x1.size()
    C1 = 16
    C2 = C - C1

    # 初始化大核
    plk_filter = nn.Parameter(torch.randn(C1, C1, 13, 13))  # pdim=C, pdim=16, kernel_size=13, kernel_size=13
    #用于生成共享大核
    plk_func = _geo_ensemble
    # 定义 ConvolutionalAttention
    Model = ConvolutionalAttention(pdim=C1, proj_dim_in=C2)

    plk_filter = plk_func(plk_filter) # (C1,C1,K,K)
    # 执行 ConvolutionalAttention
    out = Model(x1, plk_filter) # x1: (B,C,H,W);  plk_filter:(C1,C1,K,K);
    print(out.shape)