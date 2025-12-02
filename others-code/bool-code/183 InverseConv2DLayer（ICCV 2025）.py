import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/abs/2508.09824
    论文题目：Reverse Convolution and Its Applications to Image Restoration (ICCV 2025)
    中文题目：反向卷积及其在图像恢复中的应用 (ICCV 2025)
    讲解视频：https://www.bilibili.com/video/BV12NnezzEJ3/
        可逆卷积（Reverse Convolution）：
            实际意义：①转置卷积并不是卷积的真正逆运算问题：转置卷积当作“逆卷积”来使用，用于上采样。但从数学上看，转置卷积只是“插0再卷积”，并不能真正恢复卷积前的输入，存在不可逆问题。
                    ②传统解卷积（Deconvolution）方法的局限性：经典解卷积方法用于去模糊，效率较低，且一般只适用于灰度图/RGB图像，泛化性和适应性有限，不能作为深度网络中的通用模块。
            实现方式：主要包括频域逆卷积，通过填充边界、Softmax、可学习的λ正则化。
"""

class InverseConv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale=1, padding=2, padding_mode='circular',
                 eps=1e-5):
        super(InverseConv2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = eps

        # 输入通道必须等于输出通道（逐通道操作）
        assert self.out_channels == self.in_channels

        # 可学习的卷积核 (PSF)
        self.psf_weight = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, self.kernel_size))
        # 可学习的偏置
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))

        # 卷积核归一化，初始化为softmax分布
        self.psf_weight.data = nn.functional.softmax(
            self.psf_weight.data.view(1, self.in_channels, -1), dim=-1
        ).view(1, self.in_channels, self.kernel_size, self.kernel_size)

    def forward(self, x):
        # padding 保证卷积不丢失边界信息
        # 【填充边界】
        if self.padding > 0:
            x = nn.functional.pad(
                x,
                pad=[self.padding, self.padding, self.padding, self.padding],
                mode=self.padding_mode,
                value=0
            )
        _, _, h, w = x.shape

        # 上采样 (零插值)
        upsampled_y = self.zero_insert_upsample(x, scale=self.scale)

        # 如果需要，再用最近邻插值扩大
        if self.scale != 1:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')

        # 【频域】
        # 生成光学传递函数 (OTF)
        # 利用复共轭和幅值平方项，结合 λ 做正则化反演。
        otf = self.psf_to_otf(self.psf_weight, (h * self.scale, w * self.scale))
        otf_conj = torch.conj(otf)  # 共轭
        otf_power = torch.pow(torch.abs(otf), 2)  # 模方
        # 输入变换到频域
        FBFy = otf_conj * torch.fft.fftn(upsampled_y, dim=(-2, -1))
        #【可学习的λ正则化】
        bias_eps = torch.sigmoid(self.bias - 9.0) + self.eps
        FR = FBFy + torch.fft.fftn(bias_eps * x, dim=(-2, -1))
        # 频域乘法
        x1 = otf * FR
        FBR = torch.mean(self.block_split(x1, self.scale), dim=-1)
        invW = torch.mean(self.block_split(otf_power, self.scale), dim=-1)
        # 逆滤波公式
        invWBR = FBR.div(invW + bias_eps)
        FCBinvWBR = otf_conj * invWBR.repeat(1, 1, self.scale, self.scale)
        FX = (FR - FCBinvWBR) / bias_eps

        # 逆FFT回到时域
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        # 去掉padding区域
        if self.padding > 0:
            out = out[..., self.padding * self.scale:-self.padding * self.scale,
                      self.padding * self.scale:-self.padding * self.scale]

        return out

    def block_split(self, tensor, scale):
        """
        按 scale 将张量分割成小块。
        输入:
            tensor: (..., W, H)
            scale: 分割因子
        输出:
            (..., W/scale, H/scale, scale^2)
        """
        *leading_dims, W, H = tensor.size()
        W_s, H_s = W // scale, H // scale

        # reshape 分离 scale 维度
        blocks = tensor.view(*leading_dims, scale, W_s, scale, H_s)

        # 调整维度顺序
        permute_order = list(range(len(leading_dims))) + [len(leading_dims) + 1,
                                                          len(leading_dims) + 3,
                                                          len(leading_dims),
                                                          len(leading_dims) + 2]
        blocks = blocks.permute(*permute_order).contiguous()

        # 合并scale维度
        return blocks.view(*leading_dims, W_s, H_s, scale * scale)

    def psf_to_otf(self, psf, shape):
        """
        将点扩散函数 (PSF) 转换为光学传递函数 (OTF)。
        输入:
            psf: N x C x h x w
            shape: [H, W]
        输出:
            otf: N x C x H x W
        """
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        otf[..., :psf.shape[-2], :psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-2] / 2), -int(psf.shape[-1] / 2)), dims=(-2, -1))
        return torch.fft.fftn(otf, dim=(-2, -1))

    def zero_insert_upsample(self, x, scale=3):
        """
        零插值上采样。
        输入:
            x: N x C x H x W
        输出:
            N x C x (H*scale) x (W*scale)
        """
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * scale, x.shape[3] * scale)).type_as(x)
        z[..., ::scale, ::scale].copy_(x)
        return z


if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = InverseConv2DLayer(in_channels=32, out_channels=32, kernel_size=5, scale=2)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")