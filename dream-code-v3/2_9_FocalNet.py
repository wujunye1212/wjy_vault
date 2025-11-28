import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0.,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window  # 根据不同层, 设置卷积核的尺寸
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1] # (B,H,W,C)

        # pre linear projection
        x = self.f(x).permute(0, 3, 1, 2).contiguous() # (B,H,W,C)--self.f-->(B,H,W,2C+2)--permute->(B,2C+2,W,H);  假设focal_level==1
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level + 1), 1) # q:(B,C,W,H); ctx:(B,C,W,H); gates:(B,2,W,H)

        # context aggreation 上下文聚合
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx) # 执行多个DWConv, 以生成不同尺度上下文信息: (B,C,W,H)-->(B,C,W,H)
            ctx_all = ctx_all + ctx * self.gates[:, l:l + 1] # 将当前尺度的上下文信息与权重相乘,得到当前尺度最终特征,并累加到ctx_all: (B,C,W,H)
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True)) # 在最后一层的基础上,计算全局上下文表示: (B,C,W,H)-mean->(B,C,1,H)-mean->(B,C,1,1)
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level:] # 得到上下文聚合的总信息: (B,C,W,H)

        # normalize context 如果需要规范化的话,求一下平均值
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        self.modulator = self.h(ctx_all) # 通过1×1Conv, 获得调制器表示: (B,C,W,H)
        x_out = q * self.modulator # 调制器对每个query对象进行信息选择: (B,C,W,H) * (B,C,W,H) == (B,C,W,H)
        x_out = x_out.permute(0, 2, 3, 1).contiguous() # (B,C,W,H)--permute--> (B,W,H,C)
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # post linear porjection
        x_out = self.proj(x_out) # (B,W,H,C)-->(B,W,H,C)
        x_out = self.proj_drop(x_out) # dropout
        return x_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}'




if __name__ == '__main__':
    # (B, H, W, C)  H和W分别表示垂直方向和水平方向上的patch的数量
    X = torch.randn(1, 14,14, 64)

    # focal_factor: 控制卷积核大小的因子; focal_window:卷积核最小尺寸;  focal_level:上下文聚合中设置多少层
    Model = FocalModulation(
            dim=64, proj_drop=0., focal_window=3, focal_level=1,
            use_postln_in_modulation=False, normalize_modulator=False
        )
    out = Model(X)
    print(out.shape)