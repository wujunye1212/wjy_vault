import torch.nn.functional as F
import torch
import torch.nn as nn
from quan_w import Conv2dLSQ
"""
    一共三个文件 都要有！！！
    _quan_base_plus.py
    quan_w.py 
"""

"""
    论文地址：https://arxiv.org/pdf/2501.13492
    论文题目：QUANTIZED SPIKE-DRIVEN TRANSFORMER (ICLR 2025)
    中文题目：量化脉冲驱动Transformer (ICLR 2025)
    讲解视频：https://www.bilibili.com/video/BV14tKsepEeN/
        量化脉冲驱动Transformer（QSD-Transformer)：
            实际意义：脉冲神经（SNNs，第三代神经网络）因生物合理性、稀疏脉冲驱动通信和低功耗，在实现高效计算智能方面极具潜力，但任务精度较低。
            实现方式：IE-LIF神经元使用多位脉冲，这样可以更好地保留信息，减少量化误差，有助于模型学习到更准确的特征。（原文中侧重于数学公式的推理，这里只是简要概述功能）
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

# 定义一个批归一化（Batch Normalization）和填充（Padding）的自定义层
class BNAndPadLayer(nn.Module):
    def __init__(
            self,
            pad_pixels,           # 填充的像素数
            num_features,         # 通道数（特征数）
            eps=1e-5,             # 防止除零的小值
            momentum=0.1,         # 动量，用于计算移动平均
            affine=True,          # 是否使用可学习的缩放和偏移参数
            track_running_stats=True,  # 是否跟踪运行时的均值和方差
    ):
        super(BNAndPadLayer, self).__init__()
        # 定义2D批归一化层
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels  # 保存填充像素数

    def forward(self, input):
        # 对输入进行批归一化
        output = self.bn(input)
        if self.pad_pixels > 0:  # 如果需要填充
            if self.bn.affine:  # 如果批归一化层使用了可学习参数
                pad_values = (
                        self.bn.bias.detach()  # 偏置值
                        - self.bn.running_mean  # 移动平均的均值
                        * self.bn.weight.detach()  # 缩放值
                        / torch.sqrt(self.bn.running_var + self.bn.eps)  # 移动平均的标准差
                )
            else:  # 如果没有可学习参数
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            # 使用F.pad进行填充，填充的顺序是[左, 右, 上, 下]
            output = F.pad(output, [self.pad_pixels] * 4)
            # 将填充区域填充为计算得到的pad_values
            pad_values = pad_values.view(1, -1, 1, 1)  # 调整形状
            output[:, :, 0: self.pad_pixels, :] = pad_values  # 填充顶部
            output[:, :, -self.pad_pixels:, :] = pad_values  # 填充底部
            output[:, :, :, 0: self.pad_pixels] = pad_values  # 填充左侧
            output[:, :, :, -self.pad_pixels:] = pad_values  # 填充右侧
        return output

    # 定义属性以方便访问批归一化的参数
    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

# 定义一个带有可重参数化卷积（RepConv）的模块
class RepConv(nn.Module):
    def __init__(
        self,
        in_channel,  # 输入通道数
        out_channel,  # 输出通道数
        bias=False,  # 是否使用偏置
    ):
        super().__init__()
        # 定义1x1卷积
        conv1x1 = Conv2dLSQ(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        # 定义批归一化和填充层
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        # 定义3x3卷积的序列
        conv3x3 = nn.Sequential(
            Conv2dLSQ(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            Conv2dLSQ(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        # 将所有模块组合成一个顺序容器
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        # 前向传播
        return self.body(x)

# 定义一个自定义的ReLU激活函数，限制输出范围为[0, thre]
class ReLUX(nn.Module):
    def __init__(self, thre=8):  # thre是输出的上限值
        super(ReLUX, self).__init__()
        self.thre = thre  # 保存上限值

    def forward(self, input):
        # 使用torch.clamp将输入限制在[0, thre]之间
        return torch.clamp(input, 0, self.thre)

# 定义一个ReLUX实例，thre=4
relu4 = ReLUX(thre=4)

#----------------------------------------------------------------------
# 定义一个多脉冲激活函数，继承自torch.autograd.Function
class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens):  # 前向传播
        ctx.save_for_backward(input)  # 保存输入以备反向传播使用
        ctx.lens = lens  # 保存脉冲长度
        return torch.floor(relu4(input) + 0.5)  # 应用ReLUX并进行四舍五入

    @staticmethod
    def backward(ctx, grad_output):  # 反向传播
        input, = ctx.saved_tensors  # 获取保存的输入
        grad_input = grad_output.clone()  # 克隆梯度输出
        temp1 = 0 < input  # 判断输入是否大于0
        temp2 = input < ctx.lens  # 判断输入是否小于脉冲长度
        return grad_input * temp1.float() * temp2.float(), None  # 返回梯度

# 定义一个多脉冲激活模块，封装multispike
class Multispike(nn.Module):
    def __init__(self, lens=4):  # lens是脉冲长度
        super().__init__()
        self.lens = lens  # 保存脉冲长度
        self.spike = multispike  # 保存激活函数

    def forward(self, inputs):
        # 调用激活函数并归一化
        return self.spike.apply(4 * inputs, self.lens) / 4

# 定义一个多脉冲注意力模块，与Multispike类似，但归一化因子不同
class Multispike_att(nn.Module):
    def __init__(self, lens=4):
        super().__init__()
        self.lens = lens
        self.spike = multispike

    def forward(self, inputs):
        return self.spike.apply(4 * inputs, self.lens) / 2
#----------------------------------------------------------------------

# 定义一个基于多脉冲注意力机制的自注意力模块
class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
            self,
            dim,  # 输入的维度（通道数）
            num_heads=8  # 注意力头数
    ):
        super().__init__()
        # 确保通道数可以被头数整除
        assert (
                dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim  # 保存输入维度
        self.num_heads = num_heads  # 保存头数
        self.scale = 0.25  # 缩放因子

        # 定义多脉冲激活函数
        self.head_lif = Multispike()

        # 定义查询（q）、键（k）和值（v）的卷积网络
        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        # 定义查询、键和值的多脉冲激活函数
        self.q_lif = Multispike()
        self.k_lif = Multispike()
        self.v_lif = Multispike()

        # 定义注意力的多脉冲激活函数
        self.attn_lif = Multispike_att()

        # 定义输出的卷积网络
        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        # 添加时间维度 T=1 不影响
        x = x.unsqueeze(0)

        # 获取张量的形状
        T, B, C, H, W = x.shape  # T: 时间步数, B: 批量大小, C: 通道数, H: 高度, W: 宽度
        N = H * W  # 特征图的总像素数

        # 对输入应用多脉冲激活
        # torch.Size([1, 4, 64, 32, 32]) ===》 torch.Size([1, 4, 64, 32, 32])
        x = self.head_lif(x)

        # 计算查询、键和值
        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W) # torch.Size([1, 4, 64, 32, 32])
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W) # torch.Size([1, 4, 64, 32, 32])
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W) # torch.Size([1, 4, 64, 32, 32])

        # 对查询、键和值应用多脉冲激活，并调整形状
        q = self.q_lif(q).flatten(3) # torch.Size([1, 4, 64, 1024])
        q = (
            q.transpose(-1, -2)
                .reshape(T, B, N, self.num_heads, C // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
        ) # torch.Size([1, 4, 8, 1024, 8])

        k = self.k_lif(k).flatten(3) # torch.Size([1, 4, 64, 1024])
        k = (
            k.transpose(-1, -2)
                .reshape(T, B, N, self.num_heads, C // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
        ) # torch.Size([1, 4, 8, 1024, 8])

        v = self.v_lif(v).flatten(3) # torch.Size([1, 4, 64, 1024])
        v = (
            v.transpose(-1, -2)
                .reshape(T, B, N, self.num_heads, C // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
        ) # torch.Size([1, 4, 8, 1024, 8])

        # 计算注意力分数
        x = k.transpose(-2, -1) @ v  # 键和值的点积
        x = (q @ x) * self.scale  # 查询和键值点积的缩放

        # 调整形状并应用注意力激活
        # torch.Size([1, 4, 64, 1024])
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous() # torch.Size([1, 4, 64, 32, 32])
        x = self.attn_lif(x).reshape(T, B, C, H, W) # torch.Size([4, 64, 32, 32])
        x = x.reshape(T, B, C, H, W)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, H, W)

        # 去掉时间维度
        x = x.squeeze(0)
        return x

if __name__ == "__main__":

    B, C, H, W = 4, 64, 32, 32  # 定义输入张量的形状
    input_tensor = torch.randn(B, C, H, W)  # 随机生成输入张量
    print("输入张量形状：", input_tensor.shape)

    # 初始化模型
    # 假设输入通道数 C=64，注意力头数 num_heads=8
    model = MS_Attention_RepConv_qkv_id(dim=C, num_heads=8)
    output_tensor = model(input_tensor)
    # 输出结果的形状
    print("输出张量形状：", output_tensor.shape)
    print("公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")