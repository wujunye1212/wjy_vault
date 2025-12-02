import torch
from torch import nn
import numpy as np

"""
    论文地址：https://arxiv.org/pdf/2503.02394
    论文题目：BHViT: Binarized Hybrid Vision Transformer（CVPR 2025）
    中文题目：BHViT：二值化混合视觉Transformer
    讲解视频：https://www.bilibili.com/video/BV1WT5PzNEnc/
        多尺度二值化分组扩张卷积（Multi-scale grouped dilated convolution，MSGDC）：
            实际意义：①计算复杂度高：在特征提取阶段，特征金字塔会使特征具有较大空间分辨率，这会显著增加复杂度。
                    ②增强二值激活的表征能力：在二值化过程中，原本丰富的连续数值信息被简化为二值，不可避免会损失部分信息。
                    （如何既降低计算量，也提高精度呢？）
            实现方式：①采用分组空洞卷积，在相比普通卷积和自注意力模块，能减少参数和计算量，缓解计算压力。
                    ②在二值化的情况下，RPReLU激活函数可以更好地保留有用的信息，使得二值激活在有限的取值范围内，尽可能准确地代表原始数据特征，进而提升表征能力。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class Config:
    def __init__(self):
        self.norm_layer = nn.LayerNorm  # 归一化层类型
        self.layer_norm_eps = 1e-6  # 归一化层小量
        self.weight_bits = 1  # 权重量化位数 可以为 1，2，＜32 分别对应 BinaryQuantizer、TwnQuantizer、SymQuantizer
        self.input_bits = 1  # 输入量化位数 与self.weight_bits要对应
        self.clip_val = 1.0  # 裁剪阈值（示例值）
        self.recu = False  # 递归量化标志（示例：不启用）

# 二值量化器（自定义自动微分函数）
class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # 保存输入张量用于反向传播
        out = torch.sign(input)  # 对输入张量取符号（二值量化结果）
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors  # 获取保存的输入张量
        input = input[0]  # 提取输入张量
        # 计算左区间（[-1,0]）指示掩码
        indicate_leftmid = ((input >= -1.0) & (input <= 0)).float()
        # 计算右区间（(0,1]）指示掩码
        indicate_rightmid = ((input > 0) & (input <= 1.0)).float()
        # 计算输入梯度（根据STE近似策略）
        grad_input = (indicate_leftmid * (2 + 2 * input) + indicate_rightmid * (2 - 2 * input)) * grad_output.clone()
        return grad_input

# 三元量化器（TWN方法，自定义自动微分函数）
class TwnQuantizer(torch.autograd.Function):
    # https://arxiv.org/abs/1605.04711
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        ctx.save_for_backward(input, clip_val)  # 保存输入和裁剪值用于反向传播
        # 对输入进行裁剪（限制在[clip_val[0], clip_val[1]]区间）
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])

        if layerwise:  # 层级量化模式
            m = input.norm(p=1).div(input.nelement())  # 计算L1范数均值
            thres = 0.7 * m  # 计算阈值
            pos = (input > thres).float()  # 正区间掩码
            neg = (input < -thres).float()  # 负区间掩码
            mask = (input.abs() > thres).float()  # 有效区间掩码
            alpha = (mask * input).abs().sum() / mask.sum()  # 计算缩放因子
            result = alpha * pos - alpha * neg  # 生成三元量化结果
        else:  # 行级量化模式（适用于嵌入层/权重）
            n = input[0].nelement()  # 每行元素数量
            m = input.data.norm(p=1, dim=1).div(n)  # 按行计算L1范数均值
            thres = (0.7 * m).view(-1, 1).expand_as(input)  # 扩展阈值到输入维度
            pos = (input > thres).float()  # 正区间掩码
            neg = (input < -thres).float()  # 负区间掩码
            mask = (input.abs() > thres).float()  # 有效区间掩码
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)  # 按行计算缩放因子
            result = alpha * pos - alpha * neg  # 生成三元量化结果
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors  # 获取保存的输入和裁剪值（未裁剪的原始输入）
        grad_input = grad_output.clone()  # 复制梯度输出
        grad_input[input.ge(clip_val[1])] = 0  # 超过上界的梯度置零
        grad_input[input.le(clip_val[0])] = 0  # 低于下界的梯度置零
        return grad_input, None, None, None, None  # 返回梯度（其他参数无梯度）

# 对称量化器（自定义自动微分函数）
class SymQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        ctx.save_for_backward(input, clip_val)  # 保存输入和裁剪值用于反向传播
        # 对输入进行裁剪（限制在[clip_val[0], clip_val[1]]区间）
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])

        if layerwise:  # 层级量化模式
            max_input = torch.max(torch.abs(input)).expand_as(input)  # 全局最大绝对值
        else:  # 非层及量化模式（按维度计算）
            if input.ndimension() <= 3:  # 低维张量处理
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:  # 四维张量处理（如卷积特征图）
                tmp = input.view(input.shape[0], input.shape[1], -1)  # 展平空间维度
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError("不支持的张量维度")  # 维度错误提示

        s = (2 ** (num_bits - 1) - 1) / max_input  # 计算缩放因子
        output = torch.round(input * s).div(s)  # 量化并反缩放
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors  # 获取保存的输入和裁剪值（未裁剪的原始输入）
        grad_input = grad_output.clone()  # 复制梯度输出
        grad_input[input.ge(clip_val[1])] = 0  # 超过上界的梯度置零
        grad_input[input.le(clip_val[0])] = 0  # 低于下界的梯度置零
        return grad_input, None, None, None, None  # 返回梯度（其他参数无梯度）

# 量化卷积层（继承自标准Conv2d）
class QuantizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, bias=True, config=None):
        super(QuantizeConv2d, self).__init__(*kargs, bias=bias)  # 调用父类构造
        self.weight_bits = config.weight_bits  # 权重量化位数
        self.input_bits = config.input_bits  # 输入量化位数
        self.recu = config.recu  # 递归量化标志

        # 初始化权重量化器
        if self.weight_bits == 1:  # 二值量化
            self.weight_quantizer = BinaryQuantizer
        elif self.weight_bits == 2:  # 三元量化
            self.weight_quantizer = TwnQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))  # 注册裁剪值缓冲
        elif self.weight_bits < 32:  # 低位对称量化（<32位）
            self.weight_quantizer = SymQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))  # 注册裁剪值缓冲

        # 初始化激活量化器
        if self.input_bits == 1:  # 二值量化
            self.act_quantizer = BinaryQuantizer
        elif self.input_bits == 2:  # 三元量化
            self.act_quantizer = TwnQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))  # 注册裁剪值缓冲
        elif self.input_bits < 32:  # 低位对称量化（<32位）
            self.act_quantizer = SymQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))  # 注册裁剪值缓冲

    def forward(self, input, recu=False):
        # 处理权重量化
        if self.weight_bits == 1:  # 二值权重量化
            real_weights = self.weight  # 原始权重
            # 计算通道维度的缩放因子（均值）
            scaling_factor = torch.mean(
                torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True), dim=1,
                keepdim=True)
            real_weights = real_weights - real_weights.mean([1, 2, 3], keepdim=True)  # 权重去均值

            if recu:  # 递归量化模式
                # 归一化权重（根据方差）
                real_weights = real_weights / (
                            torch.sqrt(real_weights.var([1, 2, 3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
                EW = torch.mean(torch.abs(real_weights))  # 权重绝对值均值
                Q_tau = (- EW * np.log(2 - 2 * 0.92)).detach().cpu().item()  # 计算阈值
                scaling_factor = scaling_factor.detach()  # 缩放因子去梯度
                binary_weights_no_grad = scaling_factor * torch.sign(real_weights)  # 无梯度的二值权重
                cliped_weights = torch.clamp(real_weights, -Q_tau, Q_tau)  # 裁剪后的权重
                weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights  # 结合梯度的权重
            else:  # 非递归模式
                scaling_factor = scaling_factor.detach()  # 缩放因子去梯度
                binary_weights_no_grad = scaling_factor * torch.sign(real_weights)  # 无梯度的二值权重
                cliped_weights = torch.clamp(real_weights, -1.0, 1.0)  # 裁剪后的权重
                weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights  # 结合梯度的权重
        elif self.weight_bits < 32:  # 低位量化（非二值）
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)  # 应用量化器
        else:  # 全精度（32位）
            weight = self.weight

        # 处理输入量化（仅当输入位数为1时）
        if self.input_bits == 1:
            input = self.act_quantizer.apply(input)

        # 执行卷积操作
        out = nn.functional.conv2d(
            input, weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        # 添加偏置（如果存在）
        if not self.bias is None:
            out = out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return out

# 带可学习偏移的PReLU激活函数
class RPReLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.move1 = nn.Parameter(torch.zeros(hidden_size))  # 输入偏移参数
        self.prelu = nn.PReLU(hidden_size)  # PReLU激活层
        self.move2 = nn.Parameter(torch.zeros(hidden_size))  # 输出偏移参数

    def forward(self, x):
        # 输入偏移 -> 转置维度 -> PReLU激活 -> 转置回原维度 -> 输出偏移
        out = self.prelu((x - self.move1).transpose(-1, -2)).transpose(-1, -2) + self.move2
        return out

# 可学习偏置层
class LearnableBiasnn(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBiasnn, self).__init__()
        # 初始化可学习偏置参数（形状为[1, out_chn, 1, 1]）
        self.bias = nn.Parameter(torch.zeros([1, out_chn, 1, 1]), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)  # 将偏置扩展到输入维度并相加
        return out

class MultiScale_Grouped_Dilated_Convolution(nn.Module):
    def __init__(self, in_chn, config, dilation1=1, dilation2=3, dilation3=5, kernel_size=3, stride=1, padding='same'):
        super(MultiScale_Grouped_Dilated_Convolution, self).__init__()
        self.move = LearnableBiasnn(in_chn)  # 可学习偏置层
        # 初始化不同扩张率的量化卷积层（分组数为4）
        self.cov1 = QuantizeConv2d(in_chn, in_chn, kernel_size, stride, padding, dilation1, 4, bias=True, config=config)
        self.cov2 = QuantizeConv2d(in_chn, in_chn, kernel_size, stride, padding, dilation2, 4, bias=True, config=config)
        self.cov3 = QuantizeConv2d(in_chn, in_chn, kernel_size, stride, padding, dilation3, 4, bias=True, config=config)
        self.norm = config.norm_layer(in_chn, eps=config.layer_norm_eps)  # 归一化层

        # 初始化三个带偏移的PReLU激活层
        self.act1 = RPReLU(in_chn)
        self.act2 = RPReLU(in_chn)
        self.act3 = RPReLU(in_chn)

    def forward(self, x):
        B, C, H, W = x.shape  # 获取输入张量形状
        x = self.move(x)  # 应用可学习偏置

        # 分支1：卷积->维度转置->展平->激活
        x1 = self.cov1(x)   # torch.Size([1, 32, 50, 50])
        x1 = x1.permute(0, 2, 3, 1).flatten(1, 2)   # torch.Size([1, 2500, 32])
        x1 = self.act1(x1)  # torch.Size([1, 2500, 32])

        # 分支2：卷积->维度转置->展平->激活
        x2 = self.cov2(x) # torch.Size([1, 32, 50, 50])
        x2 = x2.permute(0, 2, 3, 1).flatten(1, 2) # torch.Size([1, 2500, 32])
        x2 = self.act2(x2)  # torch.Size([1, 2500, 32])

        # 分支3：卷积->维度转置->展平->激活
        x3 = self.cov3(x) # torch.Size([1, 32, 50, 50])
        x3 = x3.permute(0, 2, 3, 1).flatten(1, 2) # torch.Size([1, 2500, 32])
        x3 = self.act3(x3) # torch.Size([1, 2500, 32])

        x = self.norm(x1 + x2 + x3)  # 合并分支并归一化 torch.Size([1, 2500, 32])
        return x.permute(0, 2, 1).view(-1, C, H, W).contiguous()

if __name__ == "__main__":
    config = Config()
    input_tensor = torch.randn(1, 32, 50, 50)
    model = MultiScale_Grouped_Dilated_Convolution(in_chn=32, config=config)
    output = model(input_tensor)
    print(f'Input size: {input_tensor.size()}')
    print(f'Output size: {output.size()}')
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")