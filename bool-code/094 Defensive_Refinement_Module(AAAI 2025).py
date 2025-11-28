import torch
import torch.nn as nn
import math
"""
    论文地址：https://arxiv.org/pdf/2412.09954
    论文题目：A2RNet: Adversarial Attack Resilient Network for Robust Infrared and Visible Image Fusion (AAAI 2025)
    中文题目：A2RNet：具有对抗攻击鲁棒性的红外和可见光图像融合网络 (AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV1YmrnYdEBB/
        防御性细化模块（Defensive Refinement Module,DRM）
             理论研究：通过增强网络对特征细化学习能力，提高了网络对噪声和特征攻击抵抗能力（抗干扰），以保证在粗细不同的融合过程中保持高质量的融合图像
"""
# 定义一个补丁嵌入类，继承自nn.Module
class PatchEmbed(nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self, x):
        # 将输入张量展平并转置
        x = x.flatten(2).transpose(1, 2)
        return x

# 定义一个补丁取消嵌入类，继承自nn.Module
class PatchUnEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
    def forward(self, x, x_size):
        # 将输入张量转置并重塑为指定大小
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x

# 定义一个自注意力机制类，继承自nn.Module
"""
    # 这个模块 和 https://www.bilibili.com/video/BV15n61YtEZX/ 很像
        高效SCC自注意力（Efficient SCC-kernel-based self-attention,ESSA）
             理论研究：通过找到映射函数将自注意力转换为线性复杂度，从而降低了计算负担。
"""
class ESSAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        b, N, C = x.shape
        qkv = self.lnqkv(x)
        qkv = torch.split(qkv, C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 计算均值并进行中心化处理
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        # 计算平方和并归一化
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)
        t1 = v
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        # 计算注意力
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)

        attn = t1 + t2
        attn = self.ln(attn)
        return attn

    # 判断两个矩阵是否相同
    def is_same_matrix(self, m1, m2):
        rows, cols = len(m1), len(m1[0])
        for i in range(rows):
            for j in range(cols):
                if m1[i][j] != m2[i][j]:
                    return False
        return True

# 定义防御性细化模块类，继承自nn.Module
class Defensive_Refinement_Module(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        # 定义卷积序列
        self.convu = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(dim * 2, dim, 1, 1, 0)
        )
        self.drop = nn.Dropout2d(0.2)
        self.norm = nn.LayerNorm(dim)
        self.attn = ESSAttn(dim)

    def forward(self, x):
        shortcut = x
        x_size = (x.shape[2], x.shape[3])
        x_embed = self.patch_embed(x) # (1, 64, 32, 32) ===>torch.Size([1, 1024, 64])

        x_embed = self.attn(self.norm(x_embed))
        # self.patch_unembed(x_embed, x_size) : torch.Size([1, 64, 32, 32]) ===
        x = self.drop(self.patch_unembed(x_embed, x_size))

        # 这一步不一样，值得学习
        # 拼接并通过卷积层
        x = torch.cat((x, shortcut), dim=1)
        x = self.convu(x)

        # 添加残差连接
        x = x + shortcut
        return x

if __name__ == '__main__':
    input = torch.rand(1, 64, 32, 32)  # 创建一个随机输入张量
    drm = Defensive_Refinement_Module(64)  # 实例化防御性细化模块
    output = drm(input)  # 获取输出
    print("DRM_input.shape:", input.shape)  # 打印输入形状
    print("DRM_output.shape:", output.shape)  # 打印输出形状
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")