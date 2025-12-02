import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
#官方原作者版本，弱点就是即插即用不强，大家使用起来不便捷
class ConvAtt(nn.Module):
    def __init__(self, dim: int, pdim: int, kernel_size: int = 13):
        super().__init__()
        self.pdim = pdim
        self.lk_size = kernel_size
        self.sk_size = 3
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(pdim, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0)
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)
        self.conv = nn.Conv2d(dim,dim,1)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.pdim, x.shape[1] - self.pdim], dim=1)

            # Dynamic Conv
            bs = x1.shape[0]
            dynamic_kernel = self.dwc_proj(x[:, :self.pdim]).reshape(-1, 1, self.sk_size, self.sk_size)
            x1_ = rearrange(x1, 'b c h w -> 1 (b c) h w')
            x1_ = F.conv2d(x1_, dynamic_kernel, stride=1, padding=self.sk_size // 2, groups=bs * self.pdim)
            x1_ = rearrange(x1_, '1 (b c) h w -> b c h w', b=bs, c=self.pdim)

            # Static LK Conv + Dynamic Conv
            x1 = F.conv2d(x1, lk_filter, stride=1, padding=self.lk_size // 2) + x1_

            x = torch.cat([x1, x2], dim=1)
            x = self.conv(x)

        else:
            # for GPU
            dynamic_kernel = self.dwc_proj(x[:, :self.pdim]).reshape(self.pdim, 1, self.sk_size, self.sk_size)
            x[:, :self.pdim] = F.conv2d(x[:, :self.pdim], lk_filter, stride=1, padding=self.lk_size // 2) \
                               + F.conv2d(x[:, :self.pdim], dynamic_kernel, stride=1, padding=self.sk_size // 2,
                                          groups=self.pdim)
            # For Mobile Conversion, uncomment the following code
            # x_1, x_2 = torch.split(x, [self.pdim, x.shape[1]-self.pdim], dim=1)
            # dynamic_kernel = self.dwc_proj(x_1).reshape(16, 1, 3, 3)
            # x_1 = F.conv2d(x_1, lk_filter, stride=1, padding=13 // 2) + F.conv2d(x_1, dynamic_kernel, stride=1, padding=1, groups=16)
            # x = torch.cat([x_1, x_2], dim=1)
        return x

    def extra_repr(self):
        return f'pdim={self.pdim}'

class ConvAtt2(nn.Module):
    def __init__(self, in_channels, att_channels=16, lk_size=13, sk_size=3, reduction=2):
        """
        :param in_channels: AIFHG输入特征图通道数
        :param att_channels: AIFHG用于注意力通道数，默认为16
        :param lk_size: AIFHG 静态大核卷积核尺寸（如图中13）
        :param sk_size: AIFHG动态卷积核尺寸（如图中3）
        :param reduction: AIFHG动态卷积中间层压缩因子
        """
        super().__init__()
        self.in_channels = in_channels
        self.att_channels = att_channels
        self.idt_channels = in_channels - att_channels
        self.lk_size = lk_size
        self.sk_size = sk_size

        # 动态卷积核生成器
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(att_channels, att_channels // reduction, 1),
            nn.GELU(),#AIFHG
            nn.Conv2d(att_channels // reduction, att_channels * sk_size * sk_size, 1)
        )
        nn.init.zeros_(self.kernel_gen[-1].weight)
        nn.init.zeros_(self.kernel_gen[-1].bias)

        # 共享静态大核卷积核：定义为参数，非卷积层
        self.lk_filter = nn.Parameter(torch.randn(att_channels, att_channels, lk_size, lk_size))
        nn.init.kaiming_normal_(self.lk_filter, mode='fan_out', nonlinearity='relu')

        # 融合层
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.att_channels + self.idt_channels, f"Input channel {C} must match att + idt ({self.att_channels} + {self.idt_channels})"

        # 通道拆分
        F_att, F_idt = torch.split(x, [self.att_channels, self.idt_channels], dim=1)

        # 生成动态卷积核 [B * att, 1, 3, 3]
        kernel = self.kernel_gen(F_att).reshape(B * self.att_channels, 1, self.sk_size, self.sk_size)

        # 动态卷积操作
        F_att_re = rearrange(F_att, 'b c h w -> 1 (b c) h w')
        out_dk = F.conv2d(F_att_re, kernel, padding=self.sk_size // 2, groups=B * self.att_channels)
        out_dk = rearrange(out_dk, '1 (b c) h w -> b c h w', b=B, c=self.att_channels)

        # 静态大核卷积
        out_lk = F.conv2d(F_att, self.lk_filter, padding=self.lk_size // 2)

        # 融合（两个卷积结果加和）
        out_att = out_lk + out_dk

        # 拼接 F_idt（保留通道）
        out = torch.cat([out_att, F_idt], dim=1)

        # 1x1 融合
        out = self.fusion(out)
        return out
# 创建一个ConvAtt实例
if __name__ == "__main__":
    # 设置输入参数
    batch_size = 2
    pdim = 48
    extra_channels = 16
    H, W = 128, 128
    # 构造输入张量：[B, pdim + extra, H, W]   [2，64，128，128]
    input = torch.randn(batch_size, pdim + extra_channels, H, W)
    # 构造静态卷积核 [pdim, pdim, K, K]
    kernel_size = 13
    lk_filter = torch.randn(extra_channels, extra_channels, kernel_size, kernel_size)
    # 实例化模块
    model = ConvAtt(dim=pdim + extra_channels,pdim=extra_channels)
    model.train()  # 启用训练模式（否则会走 inference 分支）
    # 前向传播
    output = model(input, lk_filter)
    print("原来的ConvAtt_输入张量形状:", input.shape)
    print("原来的ConvAtt_输出张量形状:", output.shape)

    #建议群里的小伙伴使用我对这个模块重构的ConvAtt2代码，效果是一样的，更容易去即插即用去缝合使用！
    input = torch.randn(1,64,128,128)
    ConvAtt2 = ConvAtt2(in_channels=64)
    output= ConvAtt2(input)
    print("Ai缝合怪整理的ConvAtt2_输入张量形状:", input.shape)
    print("Ai缝合怪整理的ConvAtt2_输出张量形状:", output.shape)
    print("Ai缝合怪二次创新改进模块交流群商品链接在评论区，只更新顶会顶刊模块的改进！")
