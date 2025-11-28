import torch
import torch.nn as nn

"""
    论文地址：https://www.sciencedirect.com/science/article/pii/S089360802500190X
    论文题目：Dual selective fusion transformer network for hyperspectral image classification （2025 一区TOP）
    中文题目：用于高光谱图像分类的双选择性融合 Transformer 网络（2025 一区TOP）
    讲解视频：https://www.bilibili.com/video/BV1cfMvzuETm/
        核选择性融合模块（Kernel selective fusion，KSF）：
            实际意义：①固定感受野问题：不同目标需要不同尺度的上下文，固定卷积核无法适应，导致错误分类。
            实现方式：①通过多尺度卷积提取不同感受野的空间特征；②再通过注意力机制筛选关键维度，确保融合后的特征同时包含空间上下文和判别性特征。
"""

class KernelSelectiveFusionAttention(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()
        # 计算通道压缩后的维度
        d = max(dim // r, L)

        # 空间特征提取分支
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  # 分组卷积保持通道独立性
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)  # 空洞卷积扩大感受野

        # 特征压缩层
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)  # 1x1卷积降通道
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)  # 1x1卷积降通道

        # 空间注意力融合层
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)  # 融合平均/最大池化特征
        self.conv = nn.Conv2d(dim // 2, dim, 1)  # 最终输出卷积

        # 通道注意力机制
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)  # 通道注意力权重生成
        self.softmax = nn.Softmax(dim=1)  # 归一化权重

    def forward(self, x):
        batch_size = x.size(0)
        dim = x.size(1)

        # 双分支特征提取
        attn1 = self.conv0(x)  # 3x3卷积分支
        attn2 = self.conv_spatial(attn1)  # 5x5空洞卷积分支

        # 通道压缩
        attn1 = self.conv1(attn1)  # 降通道至一半 内循环的上跳跃连接
        attn2 = self.conv2(attn2)  # 降通道至一半 内循环的下跳跃连接

        # 空间特征融合 【上半部分】
        attn = torch.cat([attn1, attn2], dim=1)  # 拼接双分支特征
        avg_attn = torch.mean(attn, dim=1, keepdim=True)  # 通道平均特征
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)  # 通道最大特征
        agg = torch.cat([avg_attn, max_attn], dim=1)  # 合并空间特征

        # 通道注意力计算【下半部分】
        ch_attn1 = self.global_pool(attn)  # 全局平均池化
        z = self.fc1(ch_attn1)  # 压缩通道维度
        a_b = self.fc2(z)  # 恢复原始通道维度
        a_b = a_b.reshape(batch_size, 2, dim // 2, -1)
        a_b = self.softmax(a_b)  # 通道注意力权重归一化
        # 拆分并应用注意力权重
        a1, a2 = a_b.chunk(2, dim=1)
        a1 = a1.reshape(batch_size, dim // 2, 1, 1)
        a2 = a2.reshape(batch_size, dim // 2, 1, 1)
        w1 = a1 * agg[:, 0, :, :].unsqueeze(1)  # 应用平均特征权重
        w2 = a2 * agg[:, 0, :, :].unsqueeze(1)  # 应用最大特征权重

        # 特征融合与输出
        attn = attn1 * w1 + attn2 * w2  # 加权融合双分支
        attn = self.conv(attn).sigmoid()  # 生成空间注意力图

        return x * attn

if __name__ == '__main__':
    x = torch.randn(2,32,50,50)
    model = KernelSelectiveFusionAttention(dim=32)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")