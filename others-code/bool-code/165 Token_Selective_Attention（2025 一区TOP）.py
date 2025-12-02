import torch
from torch import nn
from einops import rearrange

"""
    论文地址：https://www.sciencedirect.com/science/article/pii/S089360802500190X
    论文题目：Dual selective fusion transformer network for hyperspectral image classification （2025 一区TOP）
    中文题目：用于高光谱图像分类的双选择性融合 Transformer 网络（2025 一区TOP）
    讲解视频：https://www.bilibili.com/video/BV13fKczWEdi/
        令牌选择性融合模块（Token selective fusion，TSF）：
            实际意义：①冗余计算与噪声干扰：传统自注意力计算所有令牌的关联，引入无关信息。
                    ②关键特征筛选不足：传统 Transformer对所有Token一视同仁，无法根据任务需求动态选择最具判别性的特征。
            实现方式：①分组卷积：将输入特征拆分为g个组，生成 Q/K/V。
                    ②Top-k 筛选：注意力矩阵中只留前 k% 高值令牌，屏蔽冗余。
                    ③加权融合：关键令牌加权后残差连接，1×1 卷积输出
"""

class Token_Selective_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, k=0.8, group_num=4):
        super(Token_Selective_Attention, self).__init__()
        self.num_heads = num_heads  # 注意力头数
        self.k = k  # 选择保留的token比例
        self.group_num = group_num  # 特征分组数
        self.dim_group = dim // group_num  # 每组特征的维度

        # 可学习参数：温度参数用于调节注意力分布
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        # 1x1卷积生成Q、K、V的基础特征
        self.qkv = nn.Conv3d(self.group_num, self.group_num * 3, kernel_size=(1, 1, 1), bias=False)

        # 深度可分离卷积增强局部特征
        self.qkv_conv = nn.Conv3d(self.group_num * 3, self.group_num * 3, kernel_size=(1, 3, 3),
                                  padding=(0, 1, 1), groups=self.group_num * 3, bias=bias)

        # 输出投影层
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 可学习的权重参数
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x):
        temp = x
        b, c, h, w = x.shape
        # 将特征按组重组: [B, C, H, W] -> [B, group_num, C//group_num, H, W]
        x = x.reshape(b, self.group_num, c // self.group_num, h, w)
        b, t, c, h, w = x.shape  # t为组数

        # 生成Q、K、V特征
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        # 重排维度并添加注意力头
        q = rearrange(q, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        k = rearrange(k, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        v = rearrange(v, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)

        # 特征归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, _, N = q.shape  # N = H*W*group_num (总token数)
        # 初始化注意力掩码
        mask = torch.zeros(b, self.num_heads, N, N, device=x.device, requires_grad=False)
        # 计算原始注意力分数
        attn = (q.transpose(-2, -1) @ k) * self.temperature

        # 选择top-k个token构建稀疏注意力图
        index = torch.topk(attn, k=int(N * self.k), dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)  # 填充掩码
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))  # 应用掩码

        # 计算注意力权重
        attn = attn.softmax(dim=-1)
        # 注意力加权聚合
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)

        # 恢复原始形状
        out = rearrange(out, 'b head c (h w t) -> b t (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.reshape(b, -1, h, w)  # 合并分组

        out = self.project_out(out)

        out = temp + out
        return out

if __name__ == '__main__':
    x = torch.randn(2,32,50,50)
    model = Token_Selective_Attention(dim=32)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")