from einops import rearrange
import torch
import torch.nn as nn
"""
    论文地址：https://arxiv.org/abs/2504.16455
    论文题目：Cross Paradigm Representation and Alignment Transformer for Image Deraining (CCF-A 2025)
    中文题目：用于图像去雨的跨范式表示与对齐 Transformer(CCF-A 2025)
    讲解视频：https://www.bilibili.com/video/BV1BwWgzNEiL/
        稀疏提示通道自注意力模块（Sparse Prompt Channel Self-Attention, SPC-SA）：
            实际意义：①传统自注意力的噪声干扰问题：在通道维度计算时，所有token（通道特征）都参与计算，大量无关或弱相关通道会引入噪声。
                    ②固定稀疏策略的适配问题：早期稀疏注意力方法通常设置固定K值来保留前K%的注意力权重，这种硬性约束无法适应复杂多变的雨水场景（例如雨丝稠密和稀疏的情况差别很大），导致模型在不同场景下表现不稳定。
            实现方式：通道自注意力 → Top-K稀疏选择 → EPGO 动态调节K → Softmax归一化 → 多头融合。
"""
class SPCS(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(SPCS, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 温度参数
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)
        # 多路注意力权重
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        # 动态门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        # 维度重排以适配多头注意力
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # 向量归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape

        gate_output = self.gate(x)
        # 动态选择Top-k
        dynamic_k = int(C * gate_output.view(b, -1).mean())

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
        attn = attn.softmax(dim=-1)

        # 多路注意力融合
        out1 = (attn @ v)
        out2 = (attn @ v)
        out3 = (attn @ v)
        out4 = (attn @ v)
        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        return out

if __name__ == '__main__':
    x = torch.randn(1, 32, 50, 50)   # 输入张量 B,C,H,W
    model = SPCS(dim=32)
    y = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {y.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")