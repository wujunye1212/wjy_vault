import torch
import math
import torch.nn as nn

"""
    论文地址：https://ieeexplore.ieee.org/abstract/document/10786275
    论文题目：CFFormer: A Cross-Fusion Transformer Framework for the Semantic Segmentation of Multisource Remote Sensing Images （TGRS 2025）
    中文题目：CFFormer：一种用于多源遥感图像语义分割的交叉融合Transformer框架（TGRS 2025）
    讲解视频：https://www.bilibili.com/video/BV18YEQz4Ert/
    多源特征融合模块（Feature Fusion Module，FFM）：
        实际意义：①跨模态信息的全局交互不足：多模态遥感图像（如光学与 SAR/DSM）的互补信息需要通过全局建模。传统方法（简单相加或拼接）仅能实现局部或浅层的特征交互，无法捕捉不同模态间的长距离依赖关系。
                ②特征冗余与噪声干扰问题：多模态数据可能存在特征冗余（如重复的背景信息）或因传感器差异的噪声，直接融合会导致模型性能下降。
        实现方式：多头交叉注意力机制 + 特征增强与融合
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

# 交叉注意力模块（核心特征交互组件）
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False, qk_scale=None):
        """
        :param dim: 输入特征的维度（通道数）
        :param num_heads: 多头注意力的头数
        :param sr_ratio: 空间缩减比例（用于降低计算量）
        :param qkv_bias: Q/K/V线性层是否使用偏置
        :param qk_scale: QK缩放因子（默认使用头维度的平方根倒数）
        """
        super(CrossAttention, self).__init__()  # 调用父类初始化

        # 维度必须能被头数整除（多头注意力的基本要求）
        assert dim % num_heads == 0, f"dim {dim} 必须能被头数 {num_heads} 整除"

        self.dim = dim  # 保存输入维度
        self.num_heads = num_heads  # 保存头数
        head_dim = dim // num_heads  # 每个头的维度（总维度/头数）
        self.scale = qk_scale or head_dim ** -0.5  # 注意力缩放因子（默认√(d_k)的倒数）

        # 定义Q/K/V线性层（注意这里的交叉注意力设计：两组Q/KV）
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)  # 第一组查询Q的线性层
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 第一组K/V的线性层（合并输出）

        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)  # 第二组查询Q的线性层
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 第二组K/V的线性层（合并输出）

        self.sr_ratio = sr_ratio  # 保存空间缩减比例

        # 当sr_ratio>1时，添加空间缩减模块（降低空间分辨率减少计算量）
        if sr_ratio > 1:
            # 第一组特征的空间缩减卷积（深度可分离卷积）
            self.sr1 = nn.Conv2d(
                dim, dim,
                kernel_size=sr_ratio + 1,  # 卷积核大小（比步长多1）
                stride=sr_ratio,  # 步长等于sr_ratio（下采样）
                padding=sr_ratio // 2,  # 填充保持尺寸对齐
                groups=dim  # 深度可分离卷积（每组处理一个通道）
            )
            self.norm1 = nn.LayerNorm(dim)  # 层归一化

            # 第二组特征的空间缩减卷积（与第一组对称）
            self.sr2 = nn.Conv2d(
                dim, dim,
                kernel_size=sr_ratio + 1,
                stride=sr_ratio,
                padding=sr_ratio // 2,
                groups=dim
            )
            self.norm2 = nn.LayerNorm(dim)

    # 前向传播函数（核心计算逻辑）
    def forward(self, x1, x2, H, W):
        """
        :param x1: 输入特征1（形状：[B, N, C]）B-批次，N-序列长度，C-通道数
        :param x2: 输入特征2（形状同x1）
        :param H: 特征图高度（用于空间还原）
        :param W: 特征图宽度（用于空间还原）
        """
        B, N, C = x1.shape  # 获取输入张量形状（B-批次，N-序列长度，C-通道数）

        # 计算查询Q（两组特征分别计算）
        # 形状变换：[B, N, C] -> [B, N, num_heads, C//num_heads] -> [B, num_heads, N, head_dim]
        q1 = self.q1(x1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = self.q2(x2).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # 处理K/V的空间缩减（当sr_ratio>1时）
        if self.sr_ratio > 1:
            # 特征1的空间缩减处理
            x_1 = x1.permute(0, 2, 1).reshape(B, C, H, W)  # 序列转特征图：[B, N, C] -> [B, C, H, W]
            x_1 = self.sr1(x_1)  # 空间缩减卷积（下采样）
            x_1 = x_1.reshape(B, C, -1).permute(0, 2, 1)  # 特征图转序列：[B, C, H', W'] -> [B, N', C]
            x_1 = self.norm1(x_1)  # 层归一化
            # 计算K1/V1（合并输出后拆分）
            # 形状变换：[B, N', 2*C] -> [B, N', 2, num_heads, head_dim] -> [2, B, num_heads, N', head_dim]
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            # 特征2的空间缩减处理（与特征1对称）
            x_2 = x2.permute(0, 2, 1).reshape(B, C, H, W)
            x_2 = self.sr2(x_2)
            x_2 = x_2.reshape(B, C, -1).permute(0, 2, 1)
            x_2 = self.norm2(x_2)
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            # 不做空间缩减时直接计算K/V
            kv1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # 拆分K和V（kv的第0维度是[K, V]）
        k1, v1 = kv1[0], kv1[1]  # K1形状：[B, num_heads, N', head_dim]，V1同
        k2, v2 = kv2[0], kv2[1]  # K2形状：[B, num_heads, N', head_dim]，V2同

        # 计算交叉注意力（q1关注k2，q2关注k1）
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale  # Q1*K2^T 并缩放（形状：[B, num_heads, N, N']）
        attn1 = attn1.softmax(dim=-1)  # 对最后一维做softmax得到注意力分数

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale  # Q2*K1^T 并缩放（形状：[B, num_heads, N, N']）
        attn2 = attn2.softmax(dim=-1)  # 注意力分数归一化

        # 应用注意力到V并恢复形状
        # 形状变换：[B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim] -> [B, N, C]
        main_out = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)  # 主输出（q1关注v2）
        aux_out = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)  # 辅助输出（q2关注v1）

        return main_out, aux_out  # 返回两组交互后的特征


# 特征交互模块（整合交叉注意力和通道变换）
class FeatureInteraction(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, sr_ratio=None, norm_layer=nn.LayerNorm):
        """
        :param dim: 输入特征维度
        :param reduction: 通道缩减比例（用于降低计算量）
        :param num_heads: 交叉注意力头数
        :param sr_ratio: 空间缩减比例
        :param norm_layer: 归一化层类型
        """
        super().__init__()

        # 通道投影层（将特征投影到缩减后的维度）
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)  # 特征1的通道投影（输出2倍缩减维度）
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)  # 特征2的通道投影（对称设计）

        # 激活函数
        self.act1 = nn.ReLU(inplace=True)  # 特征1的激活函数（inplace节省内存）
        self.act2 = nn.ReLU(inplace=True)  # 特征2的激活函数

        # 交叉注意力模块（输入维度为缩减后的维度）
        self.cross_attn = CrossAttention(
            dim // reduction,
            num_heads=num_heads,
            sr_ratio=sr_ratio
        )

        # 最终投影层（恢复原始维度）
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)  # 特征1的最终投影
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)  # 特征2的最终投影

        # 归一化层（用于残差连接后）
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2, H, W):
        # 通道投影并激活（分割为两部分：y用于直接连接，z用于交叉注意力）
        # chunk(2, dim=-1)：将最后一维分成两部分（形状：[B, N, C//reduction] * 2）
        y1, z1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)  # 特征1的投影和分割
        y2, z2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)  # 特征2的投影和分割

        # 交叉注意力交互（z1和z2作为输入）
        c1, c2 = self.cross_attn(z1, z2, H, W)  # c1: z1与z2交互结果，c2: z2与z1交互结果

        # 拼接交互结果（y保留原始信息，c添加交互信息）
        y1 = torch.cat((y1, c1), dim=-1)  # 特征1的信息拼接（形状：[B, N, 2*(C//reduction)]）
        y2 = torch.cat((y2, c2), dim=-1)  # 特征2的信息拼接

        # 最终投影并残差连接（输入特征x与投影后的y相加）
        main_out = self.norm1(x1 + self.end_proj1(y1))  # 主输出（特征1增强）
        aux_out = self.norm2(x2 + self.end_proj2(y2))  # 辅助输出（特征2增强）

        return main_out, aux_out  # 返回增强后的两组特征

# 通道嵌入模块（调整特征维度并融合）
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param reduction: 通道缩减比例
        :param norm_layer: 归一化层类型（默认批量归一化）
        """
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels  # 保存输出通道数

        # 残差连接（1x1卷积调整通道）
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # 通道嵌入序列（多级卷积调整特征）
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),  # 1x1卷积降维
            # 深度可分离卷积（保持通道数不变，增强空间特征）
            nn.Conv2d(
                out_channels // reduction,
                out_channels // reduction,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=out_channels // reduction  # 分组卷积=通道数（深度可分离）
            ),
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),  # 1x1卷积升维
            norm_layer(out_channels)  # 归一化层
        )
        self.norm = norm_layer(out_channels)  # 最终归一化层（用于残差和）

    def forward(self, x, H, W):
        """
        :param x: 输入特征（形状：[B, N, C]）
        :param H: 特征图高度
        :param W: 特征图宽度
        """
        B, N, _C = x.shape  # 获取输入形状（B-批次，N-序列长度，C-通道数）

        # 序列转特征图（用于卷积操作）
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()  # 形状：[B, C, H, W]

        # 残差路径（直接1x1卷积）
        residual = self.residual(x)  # 形状：[B, out_channels, H, W]

        # 通道嵌入路径（多级卷积）
        x = self.channel_embed(x)  # 形状：[B, out_channels, H, W]

        # 残差相加并归一化
        out = self.norm(residual + x)  # 形状：[B, out_channels, H, W]

        return out  # 返回融合后的特征

# 特征融合模块（完整流程整合）
class FeatureFusion(nn.Module):
    def __init__(self, dim, reduction=1, sr_ratio=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        """
        :param dim: 输入特征维度
        :param reduction: 通道缩减比例
        :param sr_ratio: 空间缩减比例
        :param num_heads: 交叉注意力头数
        :param norm_layer: 归一化层类型
        """
        super().__init__()

        # 交叉交互模块（特征交互的核心）
        self.cross = FeatureInteraction(
            dim=dim,
            reduction=reduction,
            num_heads=num_heads,
            sr_ratio=sr_ratio
        )

        # 通道嵌入模块（融合后调整维度）
        self.channel_emb = ChannelEmbed(
            in_channels=dim * 2,  # 输入是两组特征的拼接（维度2*dim）
            out_channels=dim,  # 输出维度恢复为dim
            reduction=reduction,
            norm_layer=norm_layer
        )

        # 初始化权重（调用自定义初始化函数）
        self.apply(self._init_weights)

    # 权重初始化函数（遵循常见的深度学习初始化策略）
    @classmethod
    def _init_weights(cls, m):
        """
        :param m: 网络模块
        """
        if isinstance(m, nn.Linear):
            # 截断正态分布初始化（防止梯度消失/爆炸）
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # 偏置初始化为0
            nn.init.constant_(m.weight, 1.0)  # 权重初始化为1（单位缩放）
        elif isinstance(m, nn.Conv2d):
            # 卷积核初始化（根据扇出计算方差）
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups  # 分组卷积时调整扇出
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))  # 正态分布初始化
            if m.bias is not None:
                m.bias.data.zero_()  # 偏置初始化为0

    def forward(self, x1, x2):
        """
        :param x1: 输入特征1（形状：[B, C, H, W]）
        :param x2: 输入特征2（形状同x1）
        """
        B, C, H, W = x1.shape  # 获取输入形状（B-批次，C-通道数，H-高度，W-宽度）

        # 特征展平（二维特征转序列，用于Transformer类操作）
        # flatten(2): 将H和W维度展平为一维（形状：[B, C, H*W]）
        # transpose(1, 2): 交换通道和序列维度（形状：[B, H*W, C]）
        x1 = x1.flatten(2).transpose(1, 2)  # 特征1展平为序列
        x2 = x2.flatten(2).transpose(1, 2)  # 特征2展平为序列

        # 交叉交互（输出增强后的两组特征）
        x1, x2 = self.cross(x1, x2, H, W)  # 形状：[B, H*W, C]（每组特征）

        # 特征拼接（合并两组增强后的特征）
        fuse = torch.cat((x1, x2), dim=-1)  # 形状：[B, H*W, 2*C]

        # 通道嵌入（调整维度并融合）
        fuse = self.channel_emb(fuse, H, W)  # 形状：[B, C, H, W]（恢复二维特征图）

        return fuse  # 返回最终融合后的特征

if __name__ == "__main__":
    x1 = torch.randn(1, 32, 50, 50)  # 形状：[B=1, C=32, H=50, W=50]
    x2 = torch.randn(1, 32, 50, 50)  # 形状：[B=1, C=32, H=50, W=50]
    fusion_module = FeatureFusion(dim=32,  reduction=1,  sr_ratio=4, num_heads=8)
    output = fusion_module(x1, x2)
    print(f"输入张量1形状: {x1.shape}")
    print(f"输入张量2形状: {x2.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")