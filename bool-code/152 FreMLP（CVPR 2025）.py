import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/abs/2412.13443
    论文题目：DarkIR: Robust Low-Light Image Restorationr（CVPR 2025）
    中文题目：DarkIR：稳健的低光图像恢复技术（CVPR 2025）
    讲解视频：
    频率域前馈网络（Feed-Forward Network in the Frequency domain，FreMLP）：
        实际意义：①增强低光图像光照：在低光环境中，存在光照不足问题，FreMLP能够在不改变图像相位，仅对幅度进行处理，校正图像的光照效果，让图像更加清晰、明亮。
                ②多分辨率下的一致性：在不同分辨率下均能保持良好增强效果。
        实现方式：傅里叶域转换  幅度增强操作  逆变换(空间域)
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        # 保存epsilon值用于反向传播
        ctx.eps = eps
        # 获取输入张量的维度：[批量大小, 通道数, 高度, 宽度]
        N, C, H, W = x.size()
        # 计算每个样本在通道维度上的均值
        mu = x.mean(1, keepdim=True)
        # 计算每个样本在通道维度上的方差
        var = (x - mu).pow(2).mean(1, keepdim=True)
        # 对输入进行归一化处理
        y = (x - mu) / (var + eps).sqrt()
        # 保存中间变量用于反向传播
        ctx.save_for_backward(y, var, weight)
        # 应用可学习的缩放和偏移参数
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的epsilon值
        eps = ctx.eps
        # 获取梯度张量的维度
        N, C, H, W = grad_output.size()
        # 获取前向传播中保存的中间变量
        y, var, weight = ctx.saved_variables
        # 计算权重缩放后的梯度
        g = grad_output * weight.view(1, C, 1, 1)
        # 计算梯度在通道维度上的均值
        mean_g = g.mean(dim=1, keepdim=True)
        # 计算梯度与归一化输入乘积的均值
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        # 计算输入的梯度
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        # 返回各个输入的梯度：输入x、权重weight、偏置bias和eps(None表示不需要梯度)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        # 注册可学习的缩放参数，初始化为1
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        # 注册可学习的偏移参数，初始化为0
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        # 设置归一化时的小常数，防止除零
        self.eps = eps

    def forward(self, x):
        # 调用自定义的LayerNorm函数进行前向传播
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class FreMLP(nn.Module):
    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        # 定义频域处理模块：1x1卷积扩展通道 -> LeakyReLU激活 -> 1x1卷积恢复通道
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        # 获取输入张量的维度
        _, _, H, W = x.shape
        # 对输入进行二维实值快速傅里叶变换，转换到频域
        x_freq = torch.fft.rfft2(x, norm='backward')
        # 计算频域表示的幅度
        mag = torch.abs(x_freq)
        # 计算频域表示的相位
        pha = torch.angle(x_freq)
        # 对幅度谱进行卷积处理
        mag = self.process1(mag)
        # 从处理后的幅度和原始相位重建频域复数表示
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        # 对重建的频域表示进行逆傅里叶变换，转回空间域
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out

class Frequency_Domain(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 实例化LayerNorm2d对输入进行归一化，注意这里c未定义，应为channels
        self.norm = LayerNorm2d(channels)
        # 实例化频域处理模块
        self.freq = FreMLP(nc=channels, expand=2)
        # 可学习的缩放参数，初始化为0
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        # 可学习的偏移参数，初始化为0
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # 保存输入作为残差连接
        A = inp
        # 对输入进行归一化
        x_step2 = self.norm(inp)  # 尺寸 [B, 2*C, H, W]
        # 在频域处理归一化后的特征
        x_freq = self.freq(x_step2)  # 尺寸 [B, C, H, W]
        # 将原始输入与频域处理结果相乘
        x = A * x_freq
        # 应用残差连接和可学习的缩放参数
        x = A + x * self.gamma
        return x

if __name__ == '__main__':
    model = Frequency_Domain(channels=32)
    input_tensor = torch.rand(1, 32, 50, 50)
    output = model(input_tensor)
    print(f'Input size: {input_tensor.size()}')
    print(f'Output size: {output.size()}')
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")