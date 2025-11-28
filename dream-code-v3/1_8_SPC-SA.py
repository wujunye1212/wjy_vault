import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


class SPCSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SPCSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.linear_0 = nn.Conv2d(dim, dim , 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),  # 输出动态 K
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x)) # (B,C,H,W)--qkv-->(B,3C,H,W)--dwconv-->(B,3C,H,W)
        q, k, v = qkv.chunk(3, dim=1) # q,k,v: (B,C,H,W)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # (B,C,H,W)-->(B,h,d,HW), C=h*d,h是注意力头的个数,d是每个头的通道数
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # (B,C,H,W)-->(B,h,d,HW)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # (B,C,H,W)-->(B,h,d,HW)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape # (B,h,d,HW)
        dynamic_k = int(C * self.gate(x).view(b, -1).mean()) # 生成动态K的大小: (B,C,H,W)--gate->(B,1,H,W)--view->(B,HW)--mean-->(1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 生成密集注意力矩阵: (B,h,d,HW) @ (B,h,HW,d) == (B,h,d,d)
        mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False) # 先生成一个全0的mask矩阵: (B,h,d,d)
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1] # 选择出K个索引: (B,h,d,dynamic_k)
        mask.scatter_(-1, index, 1.) # 根据index, 在mask矩阵的对应位置置1
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf'))) # 根据mask矩阵, 将注意力矩阵attn的前K个数值保留, 其余的置为负无穷(后续 softmax 会把它们变成 0)

        attn = attn.softmax(dim=-1) # 归一化得到注意力概率
        out1 = (attn @ v) # (B,h,d,d) @ (B,h,d,HW) == (B,h,d,HW)
        out2 = (attn @ v)
        out3 = (attn @ v)
        out4 = (attn @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4 # 相当于对结果进行加权

        x = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) # 恢复与输入相同的shape: (B,h,d,HW)--rearrange-->(B,C,H,W)

        out = self.project_out(x) # (B,C,H,W)-->(B,C,H,W)

        return out


if __name__ == '__main__':
    # (B,C,H,W)
    x1 = torch.randn(1, 64, 224, 224)
    B, C, H, W = x1.size()

    # 定义 AttentionTSSA
    Model = SPCSA(dim=C, num_heads=8, bias=True)

    # 执行 AttentionTSSA
    out = Model(x1)
    print(out.shape)