import numbers
import pywt
# pip install pywavelets==1.7.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
import torch.nn.functional as F
from torch.autograd import Function

"""
论文地址：https://arxiv.org/abs/2404.13537
论文题目：Bracketing Image Restoration and Enhancement with High-Low Frequency Decomposition（CVPR 2024）
代码讲解：https://www.bilibili.com/video/BV1XisXexEXs/
【低频特征提取】
    1、低频信息代表了图像的整体结构，所以需要较大的感受野来捕捉全局信息，因此设计了全局特征提取模块来处理低频信息，使得型能够更加有效地应对不同类型的退化，有助于恢复图像背景和轮廓。
    2、由于背景和轮廓信息占据了图像相当大的比例，具有长距离依赖性对它们的恢复是有益的。
    3、采用多尺度特征融合来考虑长距离交互，并在特征学习过程中利用Transformer建立全局上下文关系。Transformer架构摒弃了空间自注意力机制而采用通道自注意力机制。
                            这是因为空间自注意力机制会导致不可接受的计算负担。
    总体而言，全局特征提取块对输入特征进行三次下采样，在不同大小的特征图上应用通道自注意力机制，最后通过多尺度小波融合块来合并不同尺度的特征。
"""

import torch
import torch.nn as nn
from einops import rearrange  # 用于张量重排

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
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x
    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x
    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)
    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    def forward(self, x):
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(out1))
        out2 += x  # Residual connection
        return out2

class Fusion(nn.Module):
    """
        通过离散小波变换（DWT）将输入信号分解成高频和低频部分，并分别对它们进行处理。处理后，再通过逆离散小波变换（IDWT）恢复成原图
        代码讲解 https://www.bilibili.com/video/BV1BetCemEyy/
    """
    def __init__(self, in_channels, wave):
        # 初始化父类
        super(Fusion, self).__init__()
        # 初始化2D离散小波变换层
        self.dwt = DWT_2D(wave)
        # 定义一个卷积层，将in_channels*3的输入转换为in_channels
        self.convh1 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        # 定义一个残差网络处理高频部分
        self.high = ResNet(in_channels)
        # 定义另一个卷积层，将in_channels的输入转换回in_channels*3
        self.convh2 = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0, bias=True)
        # 定义一个卷积层，将in_channels*2的输入转换为in_channels
        self.convl = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        # 定义一个残差网络处理低频部分
        self.low = ResNet(in_channels)
        # 初始化2D逆离散小波变换层
        self.idwt = IDWT_2D(wave)

    def forward(self, x1, x2):
        # 获取x1张量的形状参数 torch.Size([1, 32, 32, 32])
        b, c, h, w = x1.shape

        # 在离散小波变换（DWT）中，2D图像或信号通常会被分解成四个部分，这是因为二维DWT将输入信号在水平和垂直两个方向上都进行了低通和高通滤波。
        x_dwt = self.dwt(x1)        # torch.Size([1, 32, 32, 32])  ===> torch.Size([1, 128, 16, 16])

        # LL(Low - Low): 低频 - 低频部分:  首先对图像进行水平方向的低通滤波，
        #                               然后再对结果进行垂直方向的低通滤波得到的。保留图像中的低频信息，即那些变化较慢的部分，比如大的结构、背景和整体亮度等。
        # LH(Low - High): 低频 - 高频部分，主要包含图像的水平边缘细节
        # HL(High - Low): 高频 - 低频部分，主要包含图像的垂直边缘细节
        # HH(High - High): 高频 - 高频部分，主要包含图像的对角线边缘细节或纹理
        ll, lh, hl, hh = x_dwt.split(c, 1)  # torch.Size([1, 32, 16, 16])

        # 将高频部分（LH, HL, HH）拼接在一起
        high = torch.cat([lh, hl, hh], 1)       # torch.Size([1, 96, 16, 16])
        # 使用convh1对高频部分进行卷积操作
        high1 = self.convh1(high)               # torch.Size([1, 32, 16, 16])
        # 通过ResNet 残差网络处理high1
        high2 = self.high(high1)                # torch.Size([1, 32, 16, 16])
        # 使用convh2将处理后的高频部分转换回原始通道数
        highf = self.convh2(high2)              # torch.Size([1, 96, 16, 16])

        # 获取ll和x2的形状参数
        b1, c1, h1, w1 = ll.shape   # torch.Size([1, 32, 16, 16])
        b2, c2, h2, w2 = x2.shape   # torch.Size([1, 32, 16, 16])

        # 如果ll的高度与x2的高度不同
        if (h1 != h2):
            # 在x2的上方添加一行零值以匹配ll的高度
            x2 = F.pad(x2, (0, 0, 1, 0), "constant", 0)

        # 将ll和调整后的x2在通道维度上拼接
        low = torch.cat([ll, x2], 1)        # torch.Size([1, 64, 16, 16])
        # 使用convl对拼接后的低频部分进行卷积操作
        low = self.convl(low)           # torch.Size([1, 32, 16, 16])
        # 通过残差网络处理low
        lowf = self.low(low)            # torch.Size([1, 32, 16, 16])

        # 将处理后的低频部分和高频部分在通道维度上拼接
        out = torch.cat((lowf, highf), 1)   # torch.Size([1, 128, 16, 16])
        # 对拼接后的结果进行2D逆离散小波变换
        out_idwt = self.idwt(out)       # torch.Size([1, 128, 16, 16]) ===> torch.Size([1, 32, 32, 32])

        # 返回最终的结果
        return out_idwt

class Channe_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Channe_Attention, self).__init__()
        # 设置注意力头的数量
        self.num_heads = num_heads
        # 初始化一个可学习参数temperature，用于缩放点积结果
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 定义qkv（查询、键、值）联合卷积层
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)  # 生成QKV三个向量的联合卷积操作
        # 定义深度可分离卷积层，用于对qkv进行进一步处理
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 定义输出投影层
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # 最终将注意力机制的结果映射回原始维度

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入张量的batch size, channels, height, width

        # 应用联合卷积和深度可分离卷积得到qkv
        qkv = self.qkv_dwconv(self.qkv(x))

        # 将qkv拆分为单独的q, k, v
        # 将qkv张量沿着通道维度分割成三部分，分别对应查询Q、键K和值V
        """
            设输入x的形状是(batch_size,channels,height,width)，经过self.qkv(x)操作后，
            生成了一个形状为(batch_size,3*channels,height,width)的张量qkv。
            这个张量包含了Q、K、V三个部分的信息，它们在通道维度上被连接在一起。
            3 个 (batch_size,channels,height,width) Q、K、V
        """
        q, k, v = qkv.chunk(3, dim=1)

        # 重新排列q, k, v以便于后续计算
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 调整形状以适应多头注意力机制
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 对q和k进行归一化
        q = torch.nn.functional.normalize(q, dim=-1)  # 在最后一个维度上进行L2归一化
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 点积后乘以temperature参数
        attn = attn.softmax(dim=-1)  # 对注意力权重应用softmax函数

        # 根据注意力权重加权求和v
        out = (attn @ v)  # 注意力权重与值向量相乘

        # 重新排列输出形状回到原来的格式
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 通过输出投影层
        out = self.project_out(out)  # 将最终的注意力结果映射回原始维度
        return out  # 返回经过注意力机制处理后的张量

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        # 初始化第一层归一化
        self.norm1 = LayerNorm(dim, LayerNorm_type)  # 归一化层，用于稳定训练过程中的输入分布
        # 初始化注意力机制
        self.attn = Channe_Attention(dim, num_heads, bias)  # 注意力机制，帮助模型学习输入数据中的重要部分
        # 初始化第二层归一化
        self.norm2 = LayerNorm(dim, LayerNorm_type)  # 另一个归一化层，在FFN之前应用
        # 初始化前馈网络
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)  # 前馈神经网络，增加模型表达能力

    def forward(self, x):
        # 应用第一个残差块：先归一化再通过注意力机制，最后加上原始输入
        x = x + self.attn(self.norm1(x))
        # 应用第二个残差块：先归一化再通过前馈网络，最后加上上一步的结果
        x = x + self.ffn(self.norm2(x))

        return x  # 返回处理后的张量

class UNet(nn.Module):
    def __init__(self, in_channels, wave):
        super(UNet, self).__init__()

        # 定义网络中的层
        # 第一个Transformer块
        self.trans1 = TransformerBlock(in_channels, 8, 2.66, False, 'WithBias')
        # 第二个Transformer块
        self.trans2 = TransformerBlock(in_channels, 8, 2.66, False, 'WithBias')
        # 第三个Transformer块
        self.trans3 = TransformerBlock(in_channels, 8, 2.66, False, 'WithBias')

        # 平均池化层，用来减小特征图尺寸
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        # 上采样层，使用Fusion方法
        self.upsample1 = Fusion(in_channels, wave)
        self.upsample2 = Fusion(in_channels, wave)

    def forward(self, x):
        x1 = x  # 保存初始输入    torch.Size([1, 32, 64, 64])

        # 经过第一个Transformer块
        x1_r = self.trans1(x)       # torch.Size([1, 32, 64, 64])

        # 下采样并经过第二个Transformer块
        x2 = self.avgpool1(x1)      # torch.Size([1, 32, 32, 32])
        x2_r = self.trans2(x2)      # torch.Size([1, 32, 32, 32])

        # 再次下采样并经过第三个Transformer块
        x3 = self.avgpool2(x2)      # torch.Size([1, 32, 16, 16])
        x3_r = self.trans3(x3)      # torch.Size([1, 32, 16, 16])

        # 上采样并将结果融合
        x4 = self.upsample1(x2_r, x3_r) # torch.Size([1, 32, 32, 32])
        out = self.upsample2(x1_r, x4)      # torch.Size([1, 32, 64, 64])

        # 确保两个张量out和x在高度维度上尺寸一致。
        b1, c1, h1, w1 = out.shape
        b2, c2, h2, w2 = x.shape
        # 如果输出的高度与输入的高度不匹配，则对输出进行填充
        if (h1 != h2):
            out = F.pad(out, (0, 0, 1, 0), "constant", 0)

        # 将最终输出与原始输入相加，形成跳跃连接
        X = out + x                 # torch.Size([1, 32, 64, 64])
        return X

if __name__ == '__main__':
    # 实例化UNet模型
    model = UNet(32, wave='haar')  # 输入通道数为32，使用的wavelet变换类型为'haar'

    # 创建随机输入数据
    input = torch.randn(1, 32, 64, 64)  # 生成形状为(1, 32, 64, 64)的随机张量作为输入

    # 模型前向传播计算
    output = model(input)  # 执行前向传播得到输出
    print('input_size:', input.size())  # 打印输入大小
    print('output_size:', output.size())  # 打印输出大小
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")