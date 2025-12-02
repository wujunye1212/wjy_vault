import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://arxiv.org/abs/2302.13800
    论文题目：Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution (ICCV 2023)
    中文题目：用于高效图像超分辨率的空间自适应特征调制（ICCV 2023）
    讲解视频：https://www.bilibili.com/video/BV1GACJYRE4g/
        空间自适应特征调制模块（Spatially-adaptive Feature Modulation ,  SAFM）
            处理逻辑：首先将归一化特征拆分为四组分别传入多尺度特征生成单元（MFGU），第一组使用3×3深度可分离卷积进行处理，
                    其余部分通过自适应最大池化进行单独采样,学习非局部特征交互和判别性特征。
"""

class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),  # 卷积，升维到 `hidden_dim`
            nn.GELU(),                           # 激活函数
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)  # 卷积，降维回 `dim`
        )

    def forward(self, x):
        return self.ccm(x)

class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels  # 特征分块的层数
        chunk_dim = dim // n_levels  # 每个分块的通道数

        # Spatial Weighting（空间加权模块）
        # 使用深度可分离卷积（groups=chunk_dim）对每个分块进行空间加权
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)]
        )

        # Feature Aggregation（特征聚合模块）
        # 使用 1x1 卷积对所有分块的特征进行融合
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation（激活函数）
        self.act = nn.GELU()

    def forward(self, x):
        # 获取输入特征的高度和宽度
        h, w = x.size()[-2:]

        # 将输入特征按通道维度均匀分割为 n_levels 个分块
        xc = x.chunk(self.n_levels, dim=1)

        # 存储每个分块的处理结果
        out = []
        for i in range(self.n_levels):
            if i > 0:
                # 对分块进行降采样，池化后的尺寸为原尺寸的 1 / (2^i)
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)  # 自适应最大池化
                s = self.mfr[i](s)  # 空间加权
                # 将降采样后的特征插值回原始尺寸
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                # 第一个分块不进行降采样，直接进行空间加权
                s = self.mfr[i](xc[i])
            out.append(s)  # 将处理后的分块特征添加到结果列表中

        # 将所有分块特征在通道维度上拼接，并通过特征聚合模块融合
        out = self.aggr(torch.cat(out, dim=1))

        # 激活函数，并与输入特征逐元素相乘（残差结构）
        out = self.act(out) * x
        return out


if __name__ == '__main__':
    input = torch.rand(1, 16, 64, 64)
    # block = SAFM(dim=16, n_levels=4)
    block = CCM(dim=16)
    output = block(input)

    print(f"Input size: {input.size()}")
    print(f"Output size: {output.size()}")

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")

