import torch
from torch import nn
from torch.nn import functional as F

"""
    论文地址：https://arxiv.org/abs/2403.10778
    论文题目：HCF-Net: Hierarchical Context Fusion Network for Infrared Small Object Detection（arxiv 2024）
    中文题目：HCF-Net：用于红外小物体检测的分层上下文融合网络（arxiv 2024）
    讲解视频：https://www.bilibili.com/video/BV1i6qxYwEFQ/
        并行化补丁感知注意模块（Parallelized Patch-Aware Attention Module，PPA）：
             作用：捕捉不同尺度和级别的特征信息，提高小物体的识别精度。
             理论支撑：多分支特征提取策略可以同时提取不同尺度的特征信息，从而更全面地了解目标形态。

    【Patch-Aware部分代码】
"""


class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim  # 输出维度
        self.patch_size = patch_size  # 补丁大小

        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)  # 第一个线性层
        self.norm = nn.LayerNorm(output_dim // 2)  # 层归一化
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)  # 第二个线性层

        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)  # 卷积层
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))  # 可训练参数
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)  # 变换矩阵

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # 调整维度顺序 B C H W  ---> B H W C
        B, H, W, C = x.shape  # 获取输入张量的形状
        P = self.patch_size  # 补丁大小

        """
            x.unfold(1, P, P)：沿着高度维度（第1维）提取大小为 P 的子块，步长为 P。这将高度 H 分成 H/P 个块。
            x.unfold(2, P, P)：沿着宽度维度（第2维）提取大小为 P 的子块，步长为 P。这将宽度 W 分成 W/P 个块。
        """
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # 将输入张量分成小块 (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P * P, C)  # 调整形状为 (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # 在最后一个维度上求平均 (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # 通过第一个 MLP (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # 通过层归一化 (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # 通过第二个 MLP (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # 计算 softmax 注意力 (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention  # 应用注意力权重 (B, H/P*W/P, output_dim)

        """
            计算:局部输出与一个可训练参数（`prompt`）之间的余弦相似度，并使用该相似度来调整输出。
        """
        # 计算余弦相似度：
        # `F.normalize(local_out, dim=-1)`: 对 `local_out` 在最后一个维度上进行归一化。
        # `F.normalize(self.prompt[None, ..., None], dim=1)`: 对 `prompt` 进行归一化，并添加两个维度以便与 `local_out` 匹配。
        # `@`: 矩阵乘法，计算 `local_out` 和 `prompt` 之间的相似度。
        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # 计算余弦相似度 B, N, 1
        # 生成掩码：`cos_sim.clamp(0, 1)`: 将余弦相似度限制在 `[0, 1]` 之间，确保相似度值不小于 0。
        mask = cos_sim.clamp(0, 1)  # 限制相似度在 [0, 1] 之间
        # 应用掩码：`local_out * mask`: 用掩码调整 `local_out`，突出与 `prompt` 更相似的部分。
        local_out = local_out * mask  # 应用掩码
        # 应用变换矩阵：`local_out @ self.top_down_transform`: 使用一个可训练的变换矩阵进一步调整输出。
        local_out = local_out @ self.top_down_transform  # 应用变换矩阵

        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # 调整形状为 (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)  # 调整维度顺序
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)  # 双线性插值恢复原尺寸
        output = self.conv(local_out)
        return output


if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 32, 32)

    # 定义输出维度和 patch 大小
    output_dim = 64
    patch_size = 4
    model = LocalGlobalAttention(output_dim=output_dim, patch_size=patch_size)
    output = model(input_tensor)

    # 打印输出形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)

    # 打印社交媒体账号信息
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")