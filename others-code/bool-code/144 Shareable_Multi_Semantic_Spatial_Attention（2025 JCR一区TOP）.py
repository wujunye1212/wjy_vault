import typing as t
import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2407.05128
    论文题目：SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention（2025 JCR一区TOP）
    中文题目：SCSA：探索空间注意力和通道注意力之间的协同效应（2025 JCR一区TOP）
    讲解视频：https://www.bilibili.com/video/BV1tH5Dz3ECf/
        共享多语义空间注意力（Shared Multi-Semantic Spatial Attention, SMSA）：
            实际意义：①多语义空间信息利用不充分问题：未能充分利用跨空间和通道的多语义信息帮助关键特征提取。
                    ②语义差异导致的信息融合问题：不同特征间存在语义差异，这会影响融合效果，对检测和分割任务不利。
                    ③空间上下文适应性问题：传统方法会压缩所有通道信息，这削弱不同特征图上下文的适应性。
            实现方式：将特征沿高度和宽度维度分解得到子特征，再通过深度可分离卷积捕捉多语义空间信息，经分组归一化和Sigmoid激活生成空间注意力图，相乘得到输出特征。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】   
"""

class Shareable_Multi_Semantic_Spatial_Attention(nn.Module):
    def __init__(
            self,
            dim: int,  # 输入特征的维度
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],  # 分组卷积核的大小
            gate_layer: str = 'sigmoid',  # 门控层的激活函数类型
    ):
        # 调用父类的构造函数
        super(Shareable_Multi_Semantic_Spatial_Attention, self).__init__()
        # 保存输入特征的维度
        self.dim = dim

        # 断言输入特征的维度应能被4整除，若不满足则抛出异常
        assert self.dim % 4 == 0, '输入特征的维度应能被4整除。'
        # 计算每个分组的通道数
        self.group_chans = group_chans = self.dim // 4

        # 定义局部深度可分离卷积层，用于处理局部特征
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        # 定义小尺寸全局深度可分离卷积层，用于捕捉小范围的全局特征
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        # 定义中尺寸全局深度可分离卷积层，用于捕捉中等范围的全局特征
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        # 定义大尺寸全局深度可分离卷积层，用于捕捉大范围的全局特征
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        # 根据门控层类型选择激活函数，用于生成注意力权重
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        # 定义用于高度方向的组归一化层，对特征进行归一化处理
        self.norm_h = nn.GroupNorm(4, dim)
        # 定义用于宽度方向的组归一化层，对特征进行归一化处理
        self.norm_w = nn.GroupNorm(4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取输入张量的批量大小、通道数、高度和宽度
        b, c, h_, w_ = x.size()
        # 在宽度维度上求平均，得到高度方向的特征
        x_h = x.mean(dim=3)
        # 将高度方向的特征按通道分组
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)

        # 在高度维度上求平均，得到宽度方向的特征
        x_w = x.mean(dim=2)
        # 将宽度方向的特征按通道分组
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        # 计算高度方向的注意力图，先将分组后的特征拼接，再进行归一化和激活操作
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        # 调整高度方向注意力图的形状，使其与输入特征的维度匹配
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        # 计算宽度方向的注意力图，先将分组后的特征拼接，再进行归一化和激活操作
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        # 调整宽度方向注意力图的形状，使其与输入特征的维度匹配
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        # 将输入特征与高度和宽度方向的注意力图相乘，增强重要区域的特征
        x = x * x_h_attn * x_w_attn

        return x

if __name__ == '__main__':
    model = Shareable_Multi_Semantic_Spatial_Attention(dim=32)
    input = torch.randn(1, 32, 50, 50)
    output = model(input)
    print(f'Input size: {input.size()}')
    print(f'Output size: {output.size()}')
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")