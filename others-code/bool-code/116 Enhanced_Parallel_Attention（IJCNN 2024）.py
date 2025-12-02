import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2305.17654
    论文题目：MixDehazeNet : Mix Structure Block For Image Dehazing Network（IJCNN 2024）
    中文题目：MixDehazeNet：用于图像去雾网络的混合结构块（IJCNN 2024）
    讲解视频：https://www.bilibili.com/video/BV17eAaeDErh/
        并行增强注意力模块（Enhanced Parallel Attention，EAP）：
            实际意义：以往方法常将通道注意力和像素注意力机制串联使用。当通道注意力先通过提取全局信息来修改原始特征，
                    然后像素注意力再从修改后的特征中提取与位置相关的信息时，无法达到全局最优。
            实现方式：EAP将简单像素注意力、通道注意力和像素注意力这三个不同的注意力模块并行设置。
                    这种并行方式使得模块能够同时从原始特征中提取与位置相关的局部信息和共享的全局信息，
                    从而实现注意力机制的全局优化，更好地去除雾霾特征（去噪）。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""
class Enhanced_Parallel_Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm2 = nn.BatchNorm2d(dim)
        # 简单像素注意力
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),  # 1x1卷积
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')  # 深度可分离卷积
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化到1x1
            nn.Conv2d(dim, dim, 1),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活函数
        )

        # 通道注意力
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化到1x1
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),  # 1x1卷积
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活函数
        )

        # 像素注意力
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),  # 1x1卷积，降维
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),  # 1x1卷积，输出单通道
            nn.Sigmoid()  # Sigmoid激活函数
        )

        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),  # 1x1卷积，升维
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim * 4, dim, 1)  # 1x1卷积，降维
        )

    def forward(self, x):
        identity = x  # 保存输入以便残差连接
        x = self.norm2(x)  # 批归一化
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)  # 拼接不同注意力机制的输出
        x = self.mlp2(x)  # 通过MLP层
        x = identity + x  # 残差连接
        return x

# 输入 B C H W, 输出 B C H W
if __name__ == '__main__':
    # 实例化模型对象
    model = Enhanced_Parallel_Attention(dim=32)
    # 生成随机输入张量
    input = torch.randn(1, 32, 64, 64)
    # 执行前向传播
    output = model(input)
    print('input_size:', input.size())  # 打印输入尺寸
    print('output_size:', output.size())  # 打印输出尺寸

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')
    print("公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
