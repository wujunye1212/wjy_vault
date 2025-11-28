import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

"""
    论文地址：https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_RMT_Retentive_Networks_Meet_Vision_Transformers_CVPR_2024_paper.pdf
    论文题目：Agent Attention: On the Integration of Softmax and Linear Attention (ECCV 2024)
    中文题目：代理注意力：关于Softmax和线性注意力的整合 (ECCV 2024)
    讲解视频：https://www.bilibili.com/video/BV13n6EYCEPA/
        代理注意力模块（Agent Attention Module , AAM）
            理论研究：Agent Attention引入少量代理令牌A，作为Q的“代理人”，收集所有特征值V的信息，并将其呈现给每个查询令牌Q。
                    这种设计使得Agent Attention具有线性的计算复杂度，同时保留了全局上下文建模的能力，在Softmax注意力的强大
                    表达能力和线性注意力的高效计算间找到了平衡，
"""
class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=49, window=14, **kwargs):
        super().__init__()
        # 基础配置参数
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 基础注意力层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # QKV线性映射
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 输出投影
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # Agent相关参数
        self.agent_num = agent_num  # Agent的数量
        self.window = window        # 特征图窗口大小

        # 特征增强模块
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                            padding=1, groups=dim)  # 深度可分离卷积

        # Agent生成模块
        pool_size = int(agent_num ** 0.5)  # 池化输出大小，例如49^0.5=7
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

        # 第一次注意力的位置编码
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))          # Agent-Token块偏置
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))  # Agent-Token行偏置
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))  # Agent-Token列偏置

        # 第二次注意力的位置编码
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))          # Token-Agent块偏置
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))  # Token-Agent行偏置
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))  # Token-Agent列偏置

        # 初始化位置编码参数
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)

    def forward(self, x):
        """
        Args:
            x: 输入特征 (num_windows*B, N, C)
            B: 批次大小
            N: Token数量 (H*W)
            C: 通道数
        """
        b, n, c = x.shape
        h = w = int(n ** 0.5)  # 特征图的高度和宽度
        num_heads = self.num_heads
        head_dim = c // num_heads

        # 1. QKV特征生成
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离Q、K、V特征

        # 2. Agent Token生成===Q
        q_t = q.reshape(b, h, w, c).permute(0, 3, 1, 2)  # 重排Q用于生成Agent
        agent_tokens = self.pool(q_t).reshape(b, c, -1).permute(0, 2, 1)  # 通过池化生成Agent表示

        # 3. 多头注意力准备
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)  # (B,H,N,D)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)  # (B,H,N,D)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)  # (B,H,N,D)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)  # (B,H,A,D)

        # 4. 第一次注意力：Agent从Token获取信息，说人话：agent_tokens作为Q
        # 第一部分位置编码
        position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        # 第二部分位置编码
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2

        # Agent-Token注意力计算
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn) # 防止过拟合
        agent_v = agent_attn @ v  # Agent获取value信息

        # 5. 第二次注意力：Token从Agent获取信息，说人话：agent_tokens作为K
        # 第一部分位置编码
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        # 第二部分位置编码
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2

        # Token-Agent注意力计算
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v  # Token获取Agent信息

        # 6. 特征增强与输出
        x = x.transpose(1, 2).reshape(b, n, c)  # 重排特征
        v_ = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n, c)  # 深度卷积增强

        # 7. 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == '__main__':
    # B，H*W,C
    X = torch.randn(1, 196, 768)
    B, N, C = X.size()
    Model = AgentAttention(dim=C, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                          agent_num=49, window=14)
    out = Model(X)
    print(out.shape)
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")