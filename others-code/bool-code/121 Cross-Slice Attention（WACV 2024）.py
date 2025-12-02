import torch
import torch.nn as nn
import torch.distributions as td

"""
    论文地址：https://openaccess.thecvf.com/content/WACV2024/papers/Hung_CSAM_A_2.5D_Cross-Slice_Attention_Module_for_Anisotropic_Volumetric_Medical_WACV_2024_paper.pdf
    论文题目：CSAM: A 2.5D Cross-Slice Attention Module for Anisotropic Volumetric Medical Image Segmentation（WACV 2024）
    中文题目：CSAM：用于各向异性体积医学图像分割的二维半交叉切片注意力模块 （WACV 2024）
    讲解视频：https://www.bilibili.com/video/BV1vF9MYWEnD/
        切片注意力模块（Cross-Slice Attention Module, CSAM）：
            实际意义：虽然，2D图像分割任务在处理体医学图像时存在明显缺陷，但医学分析通常是逐片分析图像数据，完全忽视不同切片之间信息联系。
            实现方式：①语义注意力：从整体语义层面，聚焦体数据中重要信息，忽略背景，让模型关注目标的关键特征。
                    ②位置注意力：明确每个切片上重要信息的所在位置，辅助精准分割。
                    ③切片注意力：判断哪些切片包含关键信息，抑制易造成误判特征。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

# 自定义最大值函数（支持多维度处理）
def custom_max(x, dim, keepdim=True):
    temp_x = x  # 保留原始输入
    for i in dim:  # 遍历所有需要处理的维度
        # 沿指定维度取最大值，并保留维度信息
        temp_x = torch.max(temp_x, dim=i, keepdim=True)[0]
    if not keepdim:  # 如果不需要保持维度
        temp_x = temp_x.squeeze()  # 压缩空维度
    return temp_x


# 位置注意力模块
class PositionalAttentionModule(nn.Module):
    def __init__(self):
        super(PositionalAttentionModule, self).__init__()
        # 使用7x7卷积核融合空间特征（输入通道为最大值和平均值的拼接）
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7, 7), padding=3)

    def forward(self, x):
        # B C H W
        # 沿通道和批量维度取最大值（保持维度）【B、C】
        max_x = custom_max(x, dim=(0, 1), keepdim=True)

        # 沿通道和批量维度取平均值（保持维度）
        avg_x = torch.mean(x, dim=(0, 1), keepdim=True)
        # 拼接最大值和平均值特征图（通道维度拼接）
        att = torch.cat((max_x, avg_x), dim=1)
        att = self.conv(att)  # 卷积融合特征
        att = torch.sigmoid(att)  # 生成注意力权重（0-1之间）

        return x * att  # 应用注意力权重


# 语义注意力模块
class SemanticAttentionModule(nn.Module):
    def __init__(self, in_features, reduction_rate=16):
        super(SemanticAttentionModule, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, in_features // reduction_rate),  # 降维
            nn.ReLU(),  # 激活函数
            nn.Linear(in_features // reduction_rate, in_features)  # 恢复维度
        )

    def forward(self, x):
        # B C H W
        # 沿批量、高、宽维度取最大值（不保持维度）【B、H、W】
        max_x = custom_max(x, dim=(0, 2, 3), keepdim=False).unsqueeze(0)

        # 沿批量、高、宽维度取平均值（不保持维度）
        avg_x = torch.mean(x, dim=(0, 2, 3), keepdim=False).unsqueeze(0)
        max_x = self.linear(max_x)  # 处理最大值特征
        avg_x = self.linear(avg_x)  # 处理平均值特征
        att = max_x + avg_x  # 特征融合
        # 生成注意力权重并调整维度（匹配输入维度）
        att = torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1)

        return x * att  # 应用注意力权重


# 切片注意力模块（含不确定性建模）
class SliceAttentionModule(nn.Module):
    def __init__(self, in_features, rate=4, uncertainty=True, rank=5):
        super(SliceAttentionModule, self).__init__()
        self.uncertainty = uncertainty  # 是否启用不确定性
        self.rank = rank  # 低秩分解的秩

        # 基础线性层
        self.linear = nn.Sequential(
            nn.Linear(in_features, int(in_features * rate)),  # 特征扩展
            nn.ReLU(),  # 激活函数
            nn.Linear(int(in_features * rate), in_features)  # 恢复维度
        )

        if uncertainty:
            self.non_linear = nn.ReLU()  # 非线性激活
            # 均值估计层
            self.mean = nn.Linear(in_features, in_features)
            # 对角协方差估计层（使用指数确保正值）
            self.log_diag = nn.Linear(in_features, in_features)
            # 低秩因子估计层
            self.factor = nn.Linear(in_features, in_features * rank)

    def forward(self, x):
        # B C H W
        # 沿通道、高、宽维度取最大值（保持批量维度）【C、H、W】
        max_x = custom_max(x, dim=(1, 2, 3), keepdim=False).unsqueeze(0)
        # 沿通道、高、宽维度取平均值（保持批量维度）
        avg_x = torch.mean(x, dim=(1, 2, 3), keepdim=False).unsqueeze(0)
        max_x = self.linear(max_x)  # 处理最大值特征
        avg_x = self.linear(avg_x)  # 处理平均值特征
        att = max_x + avg_x  # 特征融合

        if self.uncertainty:  # 启用不确定性建模
            temp = self.non_linear(att)  # 非线性变换
            mean = self.mean(temp)  # 获取均值
            diag = self.log_diag(temp).exp()  # 获取对角协方差（指数确保正值）
            factor = self.factor(temp)  # 获取低秩因子
            factor = factor.view(1, -1, self.rank)  # 调整形状

            # 创建低秩多元正态分布
            dist = td.LowRankMultivariateNormal(
                loc=mean,
                cov_factor=factor,
                cov_diag=diag
            )
            att = dist.sample()  # 从分布中采样

        # 生成注意力权重并调整维度（匹配输入维度）
        att = torch.sigmoid(att).squeeze().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * att  # 应用注意力权重


# 组合注意力模块（CSAM）
class CSAM(nn.Module):
    def __init__(self, num_slices, num_channels, semantic=True, positional=True, slice=True, uncertainty=True, rank=5):
        super(CSAM, self).__init__()
        self.semantic = semantic  # 是否使用语义注意力
        self.positional = positional  # 是否使用位置注意力
        self.slice = slice  # 是否使用切片注意力

        # 根据参数初始化对应模块
        if semantic:
            self.semantic_att = SemanticAttentionModule(num_channels)
        if positional:
            self.positional_att = PositionalAttentionModule()
        if slice:
            self.slice_att = SliceAttentionModule(num_slices, uncertainty=uncertainty, rank=rank)

    def forward(self, x):
        # 依次应用各个注意力模块
        if self.semantic:
            x = self.semantic_att(x)
        if self.positional:
            x = self.positional_att(x)
        if self.slice:
            x = self.slice_att(x)
        return x

if __name__ == '__main__':
    model = CSAM(num_slices=10, num_channels=64)  # 创建模型实例
    input = torch.randn(10, 64, 128, 128)  # 生成测试输入（Batchsize 10，通道64，分辨率128x128）
    output = model(input)  # 前向传播

    print('input_size:', input.size())  # 打印输入尺寸
    print('output_size:', output.size())  # 打印输出尺寸
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')
    print("公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
