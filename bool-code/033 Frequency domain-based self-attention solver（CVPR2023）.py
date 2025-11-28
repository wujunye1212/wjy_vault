import torch
import torch.nn as nn
import numbers
from einops import rearrange
'''
    论文地址：https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf
    论文题目：Effcient Frequence Domain-based Transformer for High-Quality Image Deblurring（CVPR 2023）
    中文题目：基于高效频域的高质量图像去模糊 Transformers
    讲解视频：https://www.bilibili.com/video/BV1WpyxYFEWX/
        频率域自注意力机制 Frequency Domain Self-Attention Solver（FSAS）
            首先，利用快速傅里叶变换（FFT）和卷积操作将注意力计算转移到频域，来实现高效的注意力计算，同时大大降低了计算复杂度，使得方法可以应用于更高分辨率的图像上。
'''
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# frequency domainbased self-attention solver (FSAS)
class FSAS(nn.Module):
    def __init__(self, dim, bias=False):
        super(FSAS, self).__init__()
        # 定义一个卷积层，用于将输入从dim维度转换到dim * 6维度
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)

        # 定义一个深度可分离卷积层，保持输出维度为dim * 6
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        # 定义一个投影层，用于将dim * 2维度的数据映射回dim维度
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        # 应用LayerNorm归一化，指定类型为'WithBias'
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        # 设置patch大小为8
        self.patch_size = 8

    def forward(self, x):
        # input_tensor = torch.randn(1, 64, 32, 32)
        # 将输入x通过to_hidden层得到hidden特征图
        hidden = self.to_hidden(x)      # torch.Size([1, 384, 32, 32])

        # 对hidden进行深度可分离卷积后分割成q、k、v三个部分
        # Q torch.Size([1, 128, 32, 32])
        # K torch.Size([1, 128, 32, 32])
        # V torch.Size([1, 128, 32, 32])
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        # 重新排列q k 以适应patch尺寸
        # Q P  = torch.Size([1, 128, 4, 4, 8, 8])
        # K P  = torch.Size([1, 128, 4, 4, 8, 8])
        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,patch2=self.patch_size)

        # 对q k 执行二维实数快速傅里叶变换，并且在在频域上计算q和k的乘积
        """
                torch.fft.rfft2 函数执行的是二维实数快速傅里叶变换（Real-to-Complex 2D FFT）。
                当你对一个实数值的张量进行这种变换时，输出的张量在最后一个维度上会减少到一半，然后加上1。
                这是因为对于实数输入，其傅里叶变换的结果是共轭对称的，所以只需要保存一半的数据外加零频点就可以完全恢复整个频谱.
                因此，最后一个维度从 8 减少到了 (8 // 2) + 1 = 5
        """
        q_fft = torch.fft.rfft2(q_patch.float())    # torch.Size([1, 128, 4, 4, 8, 5])
        k_fft = torch.fft.rfft2(k_patch.float())    # torch.Size([1, 128, 4, 4, 8, 5])
        out = q_fft * k_fft                         # torch.Size([1, 128, 4, 4, 8, 5])

        # 逆向傅里叶变换回到空间域
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))       # torch.Size([1, 128, 4, 4, 8, 8])
        # 重新排列out回到原始形状
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,patch2=self.patch_size)   # torch.Size([1, 128, 32, 32])
        # 对结果应用LayerNorm归一化
        out = self.norm(out)

        # 计算最终输出，即v与out的元素级相乘
        # V torch.Size([1, 128, 32, 32])
        # Out torch.Size([1, 128, 32, 32])
        output = v * out

        # 通过project_out层调整维度
        # torch.Size([1, 64, 32, 32])
        output = self.project_out(output)

        return output
if __name__ == '__main__':
    # 创建FSAS模型实例，设置输入维度为64
    model = FSAS(64)

    # 生成随机输入张量
    input_tensor = torch.randn(1, 64, 32, 32)

    # 前向传播通过模型
    output_tensor = model(input_tensor)

    # 打印输入和输出张量的形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

    # 输出社交媒体账号信息（非代码功能相关）
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")