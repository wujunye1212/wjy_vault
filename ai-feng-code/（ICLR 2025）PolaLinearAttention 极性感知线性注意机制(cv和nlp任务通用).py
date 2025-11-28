import torch
import torch.nn as nn
#论文地址;https://arxiv.org/abs/2501.15061


class PolaLinearAttention(nn.Module):
    def __init__(self, dim, hw, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.h = hw[0]
        self.w = hw[1]

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, (self.w * self.h) // (sr_ratio * sr_ratio), dim)))

    def forward(self, x):
        B, N, C = x.shape
        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, self.h, self.w)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]

        k = k + self.positional_encoding
        kernel_function = nn.ReLU()

        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * nn.functional.sigmoid(self.power)

        q = q / scale
        k = k / scale
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        q_pos = kernel_function(q) ** power
        q_neg = kernel_function(-q) ** power
        k_pos = kernel_function(k) ** power
        k_neg = kernel_function(-k) ** power

        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k = torch.cat([k_pos, k_neg], dim=-1)

        v1, v2 = torch.chunk(v, 2, dim=-1)

        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C)

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N,
                                          mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)

        v = v.reshape(B * self.num_heads, self.h, self.w, -1).permute(0, 3, 1, 2)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        x = x + v
        x = x * g

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
if __name__ == '__main__':
    #第一步初始化模块
    pla = PolaLinearAttention(32,hw=(14, 14))

    # 1.如何输入的是图片4维数据 . CV方向的小伙伴都可以拿去使用
    # 随机生成输入4维度张量：B, C, H, W
    input_img = torch.randn(2, 32, 14, 14)
    input = input_img
    input_img = input_img.reshape(2, 32, -1).transpose(-1, -2)# 将图片四维度转化成三维度张量
    # 运行前向传递
    output = pla(input_img)
    output = output.view(2, 32, 14, 14)  # 将三维度转化成图片四维度张量
    # 输出输入图片张量和输出图片张量的形状
    print("Ai缝合即插即用模块永久更新-CV_pla input_size:", input.size())
    print("Ai缝合即插即用模块永久更新-CV_pla input_size:", output.size())

    # 2.如何输入的3维数据 . NLP方向的小伙伴都可以拿去使用
    B, N, C = 2, 196, 32  # 批量大小、序列长度、特征维度
    input = torch.randn(B, N, C)
    output = pla(input)
    print('Ai缝合即插即用模块永久更新-NLP_pla input_size:', input.size())
    print('Ai缝合即插即用模块永久更新-NLP_pla output_size:', output.size())
