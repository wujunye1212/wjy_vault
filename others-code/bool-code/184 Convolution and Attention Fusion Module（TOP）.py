import torch
import torch.nn as nn
from einops import rearrange

"""
    论文地址：https://arxiv.org/pdf/2403.10067
    论文题目：Hybrid Convolutional and Attention Network for Hyperspectral Image Denoising (2024 TOP)
    中文题目：用于高光谱图像去噪的混合卷积和注意力网络 (2024 TOP)
    讲解视频：https://www.bilibili.com/video/BV1qh4CzNEce/
    卷积与注意力融合模块（Convolution and Attention Fusion Module ,CAFM）：
        实际意义：①卷积的局限性：卷积神经网络（CNN）在提取局部邻域特征（如光谱相关性、空间细节）方面表现出色，但由于感受野有限，很难捕捉长距离依赖关系。
                ②注意力机制的不足：缺少细粒度局部信息，Transformer 的注意力机制在全局建模、长程依赖捕获方面具有优势，但往往忽视了局部邻域的细节特征。
        实现方式：局部分支卷积提细节，全局分支注意力抓依赖，最后把两者相加融合，同时利用局部与全局特征进行去噪。
"""

class CAFM(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(CAFM, self).__init__()
        # 注意力头的数量
        self.num_heads = num_heads
        # 温度参数，用于缩放注意力得分（可学习）
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 一次性生成 Q、K、V 三组特征 (通道数 ×3)，卷积核大小 1×1×1
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        # 对 Q、K、V 做深度可分离卷积，增强局部建模能力
        self.qkv_dwconv = nn.Conv3d(
            dim * 3, dim * 3, kernel_size=(3, 3, 3),
            stride=1, padding=1, groups=dim * 3, bias=bias
        )
        # 输出投影，把融合后的特征映射回原始维度
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        # 一个额外的 1×1×1 卷积层，对多头特征进行线性组合
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        # 深度卷积，用于局部卷积增强，groups 保持与多头维度对齐
        self.dep_conv = nn.Conv3d(
            9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3),
            bias=True, groups=dim // self.num_heads, padding=1
        )

    def forward(self, x):
        # 输入形状 B×C×H×W
        b, c, h, w = x.shape
        # 扩展一个维度，变成 B×C×1×H×W，以适配 3D 卷积
        x = x.unsqueeze(2)

        # 先做 1×1×1 卷积得到 QKV，再用深度卷积增强局部信息
        qkv = self.qkv_dwconv(self.qkv(x))
        # 去掉扩展的维度，回到 B×3C×H×W
        qkv = qkv.squeeze(2)

        # ========== 局部卷积增强部分 ==========
        # 转换维度顺序，方便做后续操作
        f_conv = qkv.permute(0, 2, 3, 1)
        # 重新 reshape 成 (B, HW, 3*num_heads, 通道分块)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        # 对 f_all 做一个 1×1×1 卷积，得到 9 通道的局部特征
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # 把局部卷积特征 reshape 回卷积输入格式
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        # 通过深度卷积提取局部特征
        out_conv = self.dep_conv(f_conv)
        out_conv = out_conv.squeeze(2)

        # ========== 全局自注意力部分 ==========
        # 将 qkv 沿通道维度分成 Q、K、V
        q, k, v = qkv.chunk(3, dim=1)
        # 重排成 (B, num_heads, C_per_head, HW)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # 对 Q、K 做归一化，保证计算稳定
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # 计算注意力权重：Q×K^T，并乘上温度参数
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # 用注意力权重加权 V
        out = (attn @ v)
        # 还原回 (B, C, H, W) 的格式
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)

        # ========== 融合输出 ==========
        # 全局自注意力结果 + 局部卷积增强结果
        output = out + out_conv
        return output

if __name__ == '__main__':
    # 定义模型，输入通道 32，8 个注意力头
    model = CAFM(dim=32, num_heads=8)
    # 输入张量，形状 B×C×H×W
    input = torch.rand(2, 32, 50, 50)
    output = model(input)
    print(f"输入张量形状: {input.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")