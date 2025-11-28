import torch.nn.functional as F
import torch
import torch.nn as nn
"""
    论文地址：https://arxiv.org/abs/2504.16455
    论文题目：Cross Paradigm Representation and Alignment Transformer for Image Deraining (CCF-A 2025)
    中文题目：用于图像去雨的跨范式表示与对齐 Transformer(CCF-A 2025)
    讲解视频：https://www.bilibili.com/video/BV1henFzQEcb/
        空间像素细化自注意力（Spatial Pixel Refinement Self-Attention，SPR-SA）：
            实际意义：①局部细节建模不足的问题：现有的空间自注意力虽能捕获长距离像素依赖，但其计算开销非常高。
                    ②空间像素表征不够精细的问题：在细粒度局部建模方面能力有限，导致在复杂雨纹、细小纹理和局部区域的重建上表现不佳，
                                        单纯依赖全局建模容易忽略局部结构，使得背景模糊、细节缺失。
                    ③跨通道一致性不足的问题：传统空间注意力对所有通道位置一视同仁，难以刻画同一像素在不同通道下的差异性。
            实现方式：卷积提取 → 全局池化 → 像素重加权
"""
class SPRS(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x1= F.adaptive_avg_pool2d(x, (1, 1))  # 全局平均池化
        x1 = F.softmax(x1, dim=1)             # softmax归一化
        x = x1 * x                            # 加权特征
        x = self.act(x)
        x = self.conv_1(x)
        return x

if __name__ == '__main__':
    x = torch.randn(1, 32, 50, 50)
    model = SPRS(dim = 32)
    y = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {y.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")