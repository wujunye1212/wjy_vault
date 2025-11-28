import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

"""
    论文地址：https://arxiv.org/pdf/2503.06896
    论文题目：CATANet: Efﬁcient Content-Aware Token Aggregation for Lightweight Image Super-Resolution（CVPR 2025）
    中文题目：CATANet：用于轻量级图像超分辨率的高效内容感知令牌聚合网络 （CVPR 2025）
    讲解视频：https://www.bilibili.com/video/BV1w7oFYDE8p/
    局部区域自注意力 (Local-Region Self-Attention, LRSA)：
        实际意义：①增强局部特征交互：传统的注意力机制可能无法捕捉到局部区域内特征之间的联系。
                ②全局注意力的不足：全局注意力可能会在局部细节处理上有所欠缺。
        实现方式：①LRSA 模块采用了重叠补丁（窗口机制）的方式，这使得相邻局部区域之间能够有更多特征交互。
                ②LRSA 模块专注于局部区域，与处理全局信息的模块（如 TAB 模块）相互配合，形成互补关系。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】   
"""

def patch_divide(x, step, ps):
    """Crop image into patches.
    Args:
        x (Tensor): Input feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        crop_x (Tensor): Cropped patches.
        nh (int): Number of patches along the horizontal direction.
        nw (int): Number of patches along the vertical direction.
    """
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw
def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image.
    Args:
        crop_x (Tensor): Cropped patches.
        x (Tensor): Feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        ouput (Tensor): Reversed image.
    """
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output
class PreNorm(nn.Module):
    """Normalization layer.
    Args:
        dim (int): Base channels.
        fn (Module): Module after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        # 记录注意力头的数量
        self.heads = heads
        # 记录输入特征的维度
        self.dim = dim
        # 记录查询和键的维度
        self.qk_dim = qk_dim
        # 计算缩放因子，用于缩放点积注意力计算
        self.scale = qk_dim ** -0.5

        # 定义将输入映射到查询空间的线性层
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        # 定义将输入映射到键空间的线性层
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        # 定义将输入映射到值空间的线性层
        self.to_v = nn.Linear(dim, dim, bias=False)
        # 定义对注意力输出进行投影的线性层
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # 通过线性层将输入 x 分别映射到查询、键和值空间
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # 使用 map 函数和 lambda 表达式对 q、k、v 进行形状重排
        # 将形状从 (b, n, (h * d)) 重排为 (b, h, n, d)，其中 h 是注意力头的数量
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # 计算缩放点积注意力，这是 PyTorch 提供的高效实现
        out = F.scaled_dot_product_attention(q, k, v)
        # 将注意力输出的形状从 (b, h, n, d) 重排回 (b, n, (h * d))
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 通过投影线性层对重排后的注意力输出进行处理并返回
        return self.proj(out)

class LRSA(nn.Module):
    def __init__(self, dim, qk_dim=36, mlp_dim=96, heads=4):
        super().__init__()

        # 定义一个模块列表，包含注意力层和卷积前馈网络层
        self.layer = nn.ModuleList([
            # 对输入进行归一化后传入注意力模块
            PreNorm(dim, Attention(dim, heads, qk_dim)),
            # 对输入进行归一化后传入卷积前馈网络模块
            PreNorm(dim, ConvFFN(dim, mlp_dim))
        ])

    def forward(self, x):
        # 定义 patch 大小，根据数据集自行调整
        ps = 16 # 设置
        # 定义步长，比 patch 大小小 2
        step = ps - 2
        # 将输入 x 分割成多个 patch，返回分割后的 patch、行数和列数
        crop_x, nh, nw = patch_divide(x, step, ps)  # (b, n, c, ps, ps)
        # 获取分割后 patch 的形状信息，包括批次大小、patch 数量、通道数、高度和宽度
        b, n, c, ph, pw = crop_x.shape
        # 将分割后的 patch 进行形状重排，从 (b, n, c, h, w) 变为 ((b * n), (h * w), c)
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')
        # 从模块列表中获取注意力层和卷积前馈网络层
        attn, ff = self.layer
        # 对重排后的 patch 进行注意力计算，并加上残差连接
        crop_x = attn(crop_x) + crop_x
        # 将经过注意力计算后的 patch 形状重排回 (b, n, c, h, w)
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
        # 将分割后的 patch 恢复成原始输入的形状
        x = patch_reverse(crop_x, x, step, ps)
        # 获取恢复后输入的形状信息，包括批次大小、通道数、高度和宽度
        _, _, h, w = x.shape

        # 将恢复后的输入进行形状重排，从 (b, c, h, w) 变为 (b, (h * w), c)
        x = rearrange(x, 'b c h w-> b (h w) c')
        # 对重排后的输入进行卷积前馈网络计算，并加上残差连接
        x = ff(x, x_size=(h, w)) + x
        # 将经过卷积前馈网络计算后的输入形状重排回 (b, c, h, w)
        x = rearrange(x, 'b (h w) c->b c h w', h=h)

        # 返回最终处理后的输入
        return x

if __name__ == '__main__':
    input = torch.randn(1, 32,50 ,50 )
    model = LRSA(dim=32)  # 这里 dim=32 对应输入通道数
    output = model(input)
    print(f"Input size:  {input.size()}")
    print(f"Output size: {output.size()}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")