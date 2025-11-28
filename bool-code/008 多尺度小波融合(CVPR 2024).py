import pywt
# pip install pywavelets==1.7.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

"""
论文地址：https://arxiv.org/abs/2404.13537
论文题目：Bracketing Image Restoration and Enhancement with High-Low Frequency Decomposition（CVPR 2024）
讲解视频：https://www.bilibili.com/video/BV1pJsCejEzm/
    离散小波变换将大规模特征信息分离为{HH, HL, LH, LL}。
    首先将原始的小尺度信息与LL融合，然后使用逆离散小波变换将合并的信息与{HH, HL, LH}融合。
    这种方法有助于避免直接上采样小尺度特征图并与大尺度特征图合并时可能发生的结构信息丢失问题。
"""

# 定义一个DWT功能类，继承自Function
class DWT_Function(Function):
    # 定义前向传播静态方法
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        # 保证输入张量x在内存中是连续存储的
        x = x.contiguous()
        # 保存后向传播需要的参数
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        # 保存输入张量x的形状
        ctx.shape = x.shape

        # 获取输入张量x的通道数
        dim = x.shape[1]
        # 对x进行二维卷积操作，得到低频和高频分量
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        # 将四个分量按通道维度拼接起来
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        # 返回拼接后的结果
        return x

    # 定义反向传播静态方法
    @staticmethod
    def backward(ctx, dx):
        # 检查是否需要计算x的梯度
        if ctx.needs_input_grad[0]:
            # 取出前向传播时保存的权重
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            # 根据保存的形状信息重塑dx
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            # 调整dx的维度顺序并重塑
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            # 将四个小波滤波器沿零维度拼接，并重复C次以匹配输入通道数
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            # 使用转置卷积进行上采样
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        # 返回dx以及其余不需要梯度的参数
        return dx, None, None, None, None

# 定义一个二维离散小波变换模块，继承自nn.Module
class DWT_2D(nn.Module):
    # 初始化函数，接受一个小波基名称作为参数
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        # 使用pywt库创建指定的小波对象
        w = pywt.Wavelet(wave)
        # 创建分解低通和高通滤波器的Tensor
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        # 计算二维分解滤波器
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        # 注册缓冲区变量来存储滤波器
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        # 确保滤波器的数据类型为float32
        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    # 前向传播函数
    def forward(self, x):
        # 应用DWT_Function的forward方法
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

# 定义一个IDWT功能类，继承自Function
class IDWT_Function(Function):
    # 定义前向传播静态方法
    @staticmethod
    def forward(ctx, x, filters):
        # 保存后向传播需要的参数
        ctx.save_for_backward(filters)
        # 保存输入张量x的形状
        ctx.shape = x.shape

        # 根据保存的形状信息调整x的形状
        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        # 计算通道数
        C = x.shape[1]
        # 重塑x
        x = x.reshape(B, -1, H, W)
        # 重复滤波器C次以匹配输入通道数
        filters = filters.repeat(C, 1, 1, 1)
        # 使用转置卷积进行上采样
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        # 返回上采样的结果
        return x

    # 定义反向传播静态方法
    @staticmethod
    def backward(ctx, dx):
        # 检查是否需要计算x的梯度
        if ctx.needs_input_grad[0]:
            # 取出前向传播时保存的滤波器
            filters = ctx.saved_tensors
            filters = filters[0]
            # 根据保存的形状信息重塑dx
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            # 分解滤波器
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            # 对dx进行二维卷积操作，得到低频和高频分量
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            # 将四个分量按通道维度拼接起来
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        # 返回dx以及其余不需要梯度的参数
        return dx, None

# 定义一个二维逆离散小波变换模块，继承自nn.Module
class IDWT_2D(nn.Module):
    # 初始化函数，接受一个小波基名称作为参数
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        # 使用pywt库创建指定的小波对象
        w = pywt.Wavelet(wave)
        # 创建重构低通和高通滤波器的Tensor
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        # 计算二维重构滤波器
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        # 为滤波器添加额外的维度
        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        # 将四个小波滤波器沿零维度拼接
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        # 注册缓冲区变量来存储滤波器
        self.register_buffer('filters', filters)
        # 确保滤波器的数据类型为float32
        self.filters = self.filters.to(dtype=torch.float32)

    # 前向传播函数
    def forward(self, x):
        # 应用IDWT_Function的forward方法
        return IDWT_Function.apply(x, self.filters)

class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    def forward(self, x):
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(out1))
        out2 = out2 + x # Residual connection
        return out2

class Fusion(nn.Module):
    """
        通过离散小波变换（DWT）将输入信号分解成高频和低频部分，并分别对它们进行处理。处理后，再通过逆离散小波变换（IDWT）恢复成原图
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
        #                               然后再对结果进行垂直方向的低通滤波得到的。保留图像中的低频信息，即那些变化较慢的部分，
        #                               比如大的结构、背景和整体亮度等。
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

if __name__ == '__main__':

    # 实例化模型对象
    model = Fusion(32, wave='haar')

    # 生成随机输入张量
    input1 = torch.randn(1, 32, 32, 32)

    input2 = torch.randn(1, 32, 16, 16)

    # 执行前向传播
    output = model(input1,input2)

    print('output_size:',output.size())

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")