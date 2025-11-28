import torch
import torch.nn as nn
from timm.layers import DropPath

""" 
    论文地址：https://ieeexplore.ieee.org/abstract/document/10890177/ 
    论文题目：Mobile U-ViT: Revisiting large kernel and U-shaped ViT for efficient medical image segmentation (ACM MM'25) 
    中文题目：Mobile U-ViT：面向高效医学图像分割的轻量级混合网络 (ACM MM'25) 
    讲解视频： https://www.bilibili.com/video/BV1PsCuBgEjg/
    局部特征聚合块（Local Aggregation,LA）
        实际意义：①局部显著信息稀疏，往往只出现在少量像素区域：普通卷积（小感受野）容易只看到“局部无关区域”，导致模型学不到有用特征。
                ②目标边界模糊、对比度低，存在大量噪声和伪影：局部特征可能被噪声淹没，仅靠全局建模（注意力）会进一步被稀释。
        实现方式：对特征进行归一化 → 大核深度卷积提取局部结构 → 残差增强。
"""

class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 3, padding=1, groups=in_features)  # 类似 MLP 的卷积实现
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 3, padding=1, groups=in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)  # 第一层卷积
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # 第二层卷积
        x = self.drop(x)
        return x

class LocalAgg(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()

        self.pos_embed = nn.Conv2d(dim, dim, 9, padding=4, groups=dim)  # 深度卷积提取局部位置特征
        self.norm1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, 1)  # 1x1卷积用于特征融合
        self.conv2 = nn.Conv2d(dim, dim, 1)

        self.attn = nn.Conv2d(dim, dim, 9, padding=4, groups=dim)  # 局部注意力
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  # 类 MLP 卷积
        self.sg = nn.Sigmoid()  # Sigmoid 控制特征增强程度

    def forward(self, x):
        x = x + x * (self.sg(self.pos_embed(x)) - 0.5)  # 局部位置增强
        x = x + x * (self.sg(self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))) - 0.5)  # 局部注意力增强
        x = x + x * (self.sg(self.drop_path(self.mlp(self.norm2(x)))) - 0.5)  # MLP 特征增强
        return x

if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 50, 50)
    model = LocalAgg(dim=32)
    output = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")