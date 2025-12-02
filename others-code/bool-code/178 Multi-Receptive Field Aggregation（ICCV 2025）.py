import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://arxiv.org/abs/2508.09000
    论文题目：UniConvNet: Expanding Effective Receptive Field while Maintaining Asymptotically Gaussian Distribution for ConvNets of Any Scale（ICCV 2025）
    中文题目：UniConvNet：在保持渐近高斯分布的同时扩展卷积网络的有效感受野，适用于任意规模的卷积网络（ICCV 2025）
    讲解视频：https://www.bilibili.com/video/BV1hQHUztEpK/
        多感受野聚合卷积模块（Multi-Receptive Field Aggregation, MRFA）：
            实际意义：①感受野太小：小卷积核（3×3、5×5）只能看到局部区域，缺乏对远处区域的感知力，导致模型理解不完整。
                    ②分布不合理：大卷积核能扩大视野，但远处和近处特征影响“差不多大”，这破坏图像规律：越靠近像素对结果影响越强，越远的影响越弱。
                    ③计算量大：大卷积核参数量大、FLOPs高，训练和推理成本极高，很难落地使用。
            实现方式：①使用7×7、9×9、11×11卷积核，小核→精细局部特征；中核→区域上下文；大核→全局结构信息，并行组合逐步扩展有效感受野。
                    ②一个辨别器、一个放大器。
"""

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))   # 缩放 γ
        self.beta = nn.Parameter(torch.zeros(normalized_shape))   # 平移 β
        self.eps = eps
        self.channel_format = data_format
        if self.channel_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.channel_format == "channels_last":  # [N, H, W, C]
            return F.layer_norm(x, self.normalized_shape, self.gamma, self.beta, self.eps)
        elif self.channel_format == "channels_first":  # [N, C, H, W]
            chan_mean = x.mean(1, keepdim=True)                          # μ_c
            chan_var = (x - chan_mean).pow(2).mean(1, keepdim=True)      # σ_c^2
            x_norm = (x - chan_mean) / torch.sqrt(chan_var + self.eps)   # 归一化
            x_out = self.gamma[:, None, None] * x_norm + self.beta[:, None, None]
            return x_out


class MRFA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 记录总通道数 C
        self.channels_total = dim

        # ------------------------ Stage 1（使用 C/4 通道） ------------------------
        self.ln_stage1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        # 上下文分支：1×1 + DWConv-7×7
        self.ctx_branch_s1 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 4, 7, padding=3, groups=dim // 4)
        )
        # 门控分支与后整形
        self.gate_branch_s1 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.post_gate_s1 = nn.Conv2d(dim // 4, dim // 4, 1)

        # 第二个四分块预处理 + 3×3 深度卷积细化
        self.prep_quarter2 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.refine3x3_s1 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        # ------------------------ Stage 2（使用 C/2 通道） ------------------------
        self.ln_stage2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")

        self.ctx_branch_s2 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, 9, padding=4, groups=dim // 2)
        )
        self.gate_branch_s2 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.post_gate_s2 = nn.Conv2d(dim // 2, dim // 2, 1)

        self.prep_quarter3 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.refine3x3_s2 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        # 将上下文分支 (C/2) 投影到 C/4 以与第三个四分块对齐
        self.proj_ctx_to_q3 = nn.Conv2d(dim // 2, dim // 4, 1)

        # ------------------------ Stage 3（使用 3C/4 通道） ------------------------
        self.ln_stage3 = LayerNorm(dim * 3 // 4, eps=1e-6, data_format="channels_first")

        self.ctx_branch_s3 = nn.Sequential(
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 11, padding=5, groups=dim * 3 // 4)
        )
        self.gate_branch_s3 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.post_gate_s3 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)

        self.prep_quarter4 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.refine3x3_s3 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        # 将上下文分支 (3C/4) 投影到 C/4 以与第四个四分块对齐
        self.proj_ctx_to_q4 = nn.Conv2d(dim * 3 // 4, dim // 4, 1)

    def forward(self, x):
        # 假设输入 [B, C, H, W] 且 C 可被4整除
        x = self.ln_stage1(x)
        quarters = torch.split(x, self.channels_total // 4, dim=1)  # 4 个 [B, C/4, H, W]

        # ==================== Stage 1：用第 1、2 个四分块构造 C/2 特征 ====================
        ctx_feat = self.ctx_branch_s1(quarters[0])                      # 上下文分支（7×7 DWConv）
        gated_feat = ctx_feat * self.gate_branch_s1(quarters[0])       # 门控调制
        gated_feat = self.post_gate_s1(gated_feat)                     # 线性整形

        s1_q2 = self.refine3x3_s1(self.prep_quarter2(quarters[1]))    # 第二四分块预处理 + 3×3 细化
        s1_q2 = s1_q2 + ctx_feat                                      # 残差注入
        s1_out = torch.cat((s1_q2, gated_feat), dim=1)                # 通道：C/4 + C/4 = C/2

        # ==================== Stage 2：引入第 3 个四分块，得到 3C/4 特征 ====================
        s1_out = self.ln_stage2(s1_out)
        ctx_feat = self.ctx_branch_s2(s1_out)                          # 9×9 DWConv
        gated_feat = ctx_feat * self.gate_branch_s2(s1_out)
        gated_feat = self.post_gate_s2(gated_feat)

        s2_q3 = self.refine3x3_s2(self.prep_quarter3(quarters[2]))
        s2_q3 = s2_q3 + self.proj_ctx_to_q3(ctx_feat)                  # 对齐到 C/4 后相加
        s2_out = torch.cat((s2_q3, gated_feat), dim=1)                 # 通道：C/4 + C/2 = 3C/4

        # ==================== Stage 3：引入第 4 个四分块，恢复到 C 通道 ====================
        s2_out = self.ln_stage3(s2_out)
        ctx_feat = self.ctx_branch_s3(s2_out)                          # 11×11 DWConv
        gated_feat = ctx_feat * self.gate_branch_s3(s2_out)
        gated_feat = self.post_gate_s3(gated_feat)

        s3_q4 = self.refine3x3_s3(self.prep_quarter4(quarters[3]))
        s3_q4 = s3_q4 + self.proj_ctx_to_q4(ctx_feat)
        s3_out = torch.cat((s3_q4, gated_feat), dim=1)                 # 通道：C/4 + 3C/4 = C

        return s3_out


if __name__ == '__main__':
    x = torch.randn(1, 32, 50, 50)
    model = MRFA(dim=32)
    y = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {y.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
