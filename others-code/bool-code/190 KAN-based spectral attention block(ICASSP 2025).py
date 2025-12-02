import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

"""
    论文地址：https://ieeexplore.ieee.org/abstract/document/10890177/
    论文题目：KSSANet: KAN-Driven Spatial-Spectral Attention Networks for Hyperspectral Image Super-Resolution (ICASSP 2025)
    中文题目：KSSANet：基于科尔莫戈罗夫 - 阿诺德网络（KAN）的空谱注意力网络用于高光谱图像超分辨率 (ICASSP 2025)
    讲解视频：https://www.bilibili.com/video/BV1WT2MBCEjW/
    基于KAN网络的空间注意力块（KAN-based spectral attention block, KAN-SpeAB）
        实际意义：①特征建模能力不足的问题：仅用全局平均池化，容易丢失关键谱段的局部显著特征。
                ②光谱信息易失真问题：超分任务的核心需求是“空间分辨率提升” 与 “光谱保真度保持”的平衡，现有方法过度优化空间细节导致光谱特征失真，线性激活函数/MLP表达能力不足。
        实现方式：对每个光谱通道做重要性打分，并利用 KAN 局部自适应建模生成通道注意力矩阵，最后对各通道进行加权。
            Tip：说人话就是KAN-SpeAB = 用 KAN 替换 MLP 的通道注意力
"""

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()

        # 输入输出维度（类似 nn.Linear 的 in_features/out_features）
        self.in_features = in_features
        self.out_features = out_features

        # 网格点数量（对应 KAN 的刻度点，越多越精细）
        self.grid_size = grid_size

        # B 样条阶数，决定曲线平滑度
        self.spline_order = spline_order

        # 计算网格点位置间距 h
        h = (grid_range[1] - grid_range[0]) / grid_size

        # 构造 B 样条插值的网格点（多扩展 spline_order 个点，用于边界处理）
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        # 不是可训练参数，只保存为 buffer
        self.register_buffer("grid", grid)

        # KAN 的线性权重（基础线性层 W）
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # 存储 spline 系数的权重（每个输入特征对应不同样条系数）
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        # 是否独立缩放 spline 权重
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        # 控制 KAN 初始化噪声和缩放比例
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline

        # 基础激活函数（默认 SiLU）
        self.base_activation = base_activation()

        # 融合 adaptive grid 与 uniform grid 的比例
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化基础线性层权重
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            # 初始化 spline 权重噪声（提高逼近能力）
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )

            # 将曲线噪声转为 spline 系数
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )

            # 可选：独立缩放 spline 权重
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        # x: (batch, in_features)
        assert x.dim() == 2 and x.size(1) == self.in_features

        # grid: (in_features, grid_size + 2*spline_order + 1)
        grid: torch.Tensor = self.grid

        # 扩展维度，使 x 与 grid 可以广播计算
        x = x.unsqueeze(-1)

        # 计算 0 阶 B 样条基函数（位于区间就为1，不在则为0）
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # 递推计算高阶 B 样条基函数
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]
                + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )

        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        # 将目标曲线 y 转换为 spline 系数
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        # A: B 样条矩阵
        A = self.b_splines(x).transpose(0, 1)

        # B: 目标曲线点
        B = y.transpose(0, 1)

        # 最小二乘求解 spline 系数
        solution = torch.linalg.lstsq(A, B).solution

        # (out_features, in_features, grid_size + spline_order)
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        # 对 spline 权重缩放（可调节）
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features

        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # 基础线性输出：类似 nn.Linear
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # 样条输出：基于 B 样条插值得到非线性映射
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )

        # 加法融合：KAN = Linear + B_SPLINE 非线性
        output = base_output + spline_output

        return output.reshape(*original_shape[:-1], self.out_features)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        # 自适应更新 grid 点，使其更适配数据分布
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        # 数据排序（用于采样 grid 节点）
        x_sorted = torch.sort(x, dim=0)[0]

        # adaptive grid: 按数据分布采样网格点
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]

        # uniform grid: 均匀分布
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # 融合 adaptive 与 uniform
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        # 填补边界
        grid = torch.concatenate(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        # 更新 grid 与 spline 系数
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # 对 KAN 正则化，使 spline 学习更有选择性（稀疏 & 低熵）
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()

        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())

        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class SpeKAN(nn.Module):
    def __init__(self, dim, scale):
        super(SpeKAN, self).__init__()

        self.dim = dim
        self.scale = scale

        # 分别构建 KAN 的 avg/max 两条 MLP-like 路径
        self.avg_kan01 = KANLinear(dim, dim)
        self.avg_kan02 = KANLinear(dim, dim)
        self.max_kan01 = KANLinear(dim, dim)
        self.max_kan02 = KANLinear(dim, dim)

        # 将 avg 与 max 拼接后，再做一次 KAN 映射
        self.sum_kan = KANLinear(dim * 2, dim)

    def forward(self, x):
        # 平均池化得到全局特征 (b, c, 1, 1) -> (b, c)
        avg_score = F.adaptive_avg_pool2d(x, (1, 1))
        avg_score = rearrange(avg_score, 'b c 1 1 -> b c')
        avg_score = self.avg_kan01(avg_score)
        avg_score = self.avg_kan02(avg_score)

        # 最大池化得到全局特征 (b, c, 1, 1) -> (b, c)
        max_score = F.adaptive_max_pool2d(x, (1, 1))
        max_score = rearrange(max_score, 'b c 1 1 -> b c')
        max_score = self.max_kan01(max_score)
        max_score = self.max_kan02(max_score)

        # 拼接 avg + max 后再做 KAN 权重生成
        score = self.sum_kan(torch.cat([avg_score, max_score], dim=1))

        # 恢复为 (b, c, 1, 1)
        score = rearrange(score, 'b c -> b c 1 1')
        # 加权增强（残差连接 x * score + x）
        x = x * score + x
        return x

if __name__ == '__main__':
    model = SpeKAN(dim=32, scale=1.0)
    input = torch.rand(1, 32, 50, 50)
    output = model(input)
    print(f"输入张量形状: {input.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")