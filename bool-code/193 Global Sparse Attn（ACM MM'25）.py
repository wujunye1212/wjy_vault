import torch
import torch.nn as nn
from timm.layers import DropPath

""" 
    论文地址：https://ieeexplore.ieee.org/abstract/document/10890177/ 
    论文题目：Mobile U-ViT: Revisiting large kernel and U-shaped ViT for efficient medical image segmentation (ACM MM'25) 
    中文题目：Mobile U-ViT：面向高效医学图像分割的轻量级混合网络 (ACM MM'25) 
    讲解视频： https://www.bilibili.com/video/BV1MJCeBSEyS/
    全局稀疏注意力机制（Global Sparse Attention,GSA）
        实际意义：①卷积无法捕获远距离依赖：医学图像具有局部信息稀疏的特点，卷积只能看到局部Patch特征，无法理解整体结构关系。
                ②模糊边界、低对比度导致的目标区域难辨识问题：边界模糊（例如肝脏/脾脏、肿瘤/血管）噪声高、伪影多。
                ③Transformer全局建模的计算成本：直接使用标准 Transformer 的 Attention，计算复杂度为O(N²)。
        实现方式：先降维提效 → 全局交互（Transfomer） →恢复原始维度
"""

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features  # 默认输出维度和输入相同
        hidden_features = hidden_features or in_features  # 默认隐藏层维度和输入一致

        self.fc1 = nn.Linear(in_features, hidden_features)  # 线性映射 1
        self.act = act_layer()                              # GELU 激活
        self.fc2 = nn.Linear(hidden_features, out_features) # 线性映射 2
        self.drop = nn.Dropout(drop)                        # Dropout 防止过拟合

    def forward(self, x):
        x = self.fc1(x)  # 第一次线性映射
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # 第二次线性映射
        x = self.drop(x)
        return x

class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2.):
        super().__init__()

        self.num_heads = num_heads  # 多头注意力头数
        head_dim = dim // num_heads  # 每个头的维度

        self.scale = qk_scale or head_dim ** -0.5  # 注意力缩放因子
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # Q、K、V 生成层
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力 dropout
        self.proj = nn.Linear(dim, dim)         # 输出映射回原维度
        self.proj_drop = nn.Dropout(proj_drop)  # 输出 dropout

        self.sr = sr_ratio  # 下采样比例，用于稀疏注意力
        if self.sr > 1:
            self.sampler = nn.AvgPool2d(1, sr_ratio)  # 平均池化下采样
            self.LocalProp = nn.ConvTranspose2d(dim, dim, sr_ratio, stride=sr_ratio, groups=dim)  # 转置卷积恢复空间分辨率
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape  # B=批次 N=像素点个数 C=通道数

        if self.sr > 1.:
            # 将 H×W 划分为少量 token
            # 每个 token 代表一个“区域级亮度/颜色状态”
            x = x.transpose(1, 2).reshape(B, C, H, W)  # 恢复为 B C H W
            x = self.sampler(x)                       # 下采样
            x = x.flatten(2).transpose(1, 2)          # 重新展平回 Transformer 格式

        # 每个 token 与全图 token 交互
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]        # 分离 Q、K、V
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 计算注意力得分
        attn = attn.softmax(dim=-1)             # softmax 获得权重
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)  # 加权求和，得到注意力结果

        if self.sr > 1:
            # 将全局光照信息重新分发回像素级
            x = x.permute(0, 2, 1).reshape(B, C, int(H / self.sr), int(W / self.sr))  # 恢复空间形状
            x = self.LocalProp(x)  # 转置卷积扩大回原分辨率
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        x = self.proj(x)      # 输出映射
        x = self.proj_drop(x)
        return x

class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.):
        super().__init__()

        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  # 添加空间位置信息
        self.norm1 = norm_layer(dim)

        self.attn = GlobalSparseAttn(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)  # 全局稀疏注意力

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)  # 加入位置信息

        B, N, H, W = x.shape  # 获取尺寸信息

        x = x.flatten(2).transpose(1, 2)  # 转为 Transformer 格式：(B, HW, C)

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  # 自注意力
        x = x + self.drop_path(self.mlp(self.norm2(x)))         # MLP 前馈网络

        x = x.transpose(1, 2).reshape(B, N, H, W)  # reshape 回卷积格式
        return x

if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 50, 50)
    model = SelfAttn(dim=32,num_heads=8)
    output = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")