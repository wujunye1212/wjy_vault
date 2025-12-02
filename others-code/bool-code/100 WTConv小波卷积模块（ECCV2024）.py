import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
# 首先记得安装
# pip install PyWavelets
import pywt

"""
    论文地址：https://arxiv.org/abs/2407.05848
    论文题目：Wavelet Convolutions for Large Receptive Fields (ECCV 2024)
    中文题目：用于大感受野的小波卷积(ECCV 2024)
    讲解视频：https://www.bilibili.com/video/BV1P6cTeXEDZ/
        小波卷积（Wavelet Transform as Convolutions, WT-Conv）
             1、小波在信号处理领域应用广泛，近年来也被用于神经网络架构。但多数工作无法直接用于其他卷积。
             2、大核卷积核易导致过参数化，且基于傅里叶变换的方法无法学习像素间的局部交互，而小波变换在分解图像时能保留局部信息。
        相关视频：小波变换采样模块             ：https://www.bilibili.com/video/BV1Cv2WYFEnH/
                多尺度小波特征融合（CVPR2024）：https://www.bilibili.com/video/BV1pJsCejEzm/ 
"""

# 定义创建小波滤波器的函数
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    # 使用pywt库根据给定的小波类型创建一个小波对象
    w = pywt.Wavelet(wave)
    # 将分解高通滤波器反转并转换为指定类型的张量
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    # 将分解低通滤波器反转并转换为指定类型的张量
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    # 创建分解滤波器组，并通过张量操作将它们组合起来
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    # 对分解滤波器进行复制以匹配输入通道数
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    # 将重构高通滤波器反转并转换为指定类型的张量，然后再次反转
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    # 将重构低通滤波器反转并转换为指定类型的张量，然后再次反转
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    # 创建重构滤波器组，并通过张量操作将它们组合起来
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    # 对重构滤波器进行复制以匹配输出通道数
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    # 返回分解和重构滤波器
    return dec_filters, rec_filters

# 定义小波变换函数
def wavelet_transform(x, filters):
    # 获取输入张量的形状
    b, c, h, w = x.shape
    # 计算填充大小
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # 执行二维卷积以实现小波变换
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    # 调整张量形状以分离出四个子带
    x = x.reshape(b, c, 4, h // 2, w // 2)
    # 返回变换后的张量
    return x

# 定义逆小波变换函数
def inverse_wavelet_transform(x, filters):
    # 获取输入张量的形状
    b, c, _, h_half, w_half = x.shape
    # 计算填充大小
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # 调整张量形状以合并四个子带
    x = x.reshape(b, c * 4, h_half, w_half)
    # 执行二维反卷积以实现逆小波变换
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    # 返回逆变换后的张量
    return x

# 定义_ScaleModule类，用于缩放输入
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        # 设置属性
        self.dims = dims
        # 初始化权重参数
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        # 初始化偏置（此处未使用）
        self.bias = None

    # 前向传播函数
    def forward(self, x):
        # 将权重与输入张量相乘
        return torch.mul(self.weight, x)

# 定义WTConv2d类，继承自nn.Module
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        # 确保输入通道数与输出通道数相等
        assert in_channels == out_channels

        # 设置属性
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        # 创建小波滤波器
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        # 将小波滤波器设置为不可训练参数
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 部分应用函数，固定小波变换和逆变换中的滤波器参数
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        # 定义基础卷积层
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        # 定义基础缩放模块
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        # 定义多级小波卷积层
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        # 定义多级小波缩放模块
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        # 如果步长大于1，则定义步长滤波器
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            # 定义执行步长的lambda函数
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            # 否则，将do_stride设置为None
            self.do_stride = None

    def forward(self, x):
        # 初始化列表用于存储各级小波变换的结果
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        # 当前处理的是原始输入x
        curr_x_ll = x

        # 对每一级执行小波变换
        for i in range(self.wt_levels):
            # 存储当前级别的形状
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            # 如果当前级别宽度或高度是奇数，则需要进行填充
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # 执行当前级别的小波变换
            curr_x = self.wt_function(curr_x_ll)
            # 提取低频子带作为下一级别的输入
            curr_x_ll = curr_x[:, :, 0, :, :]

            # 获取当前级别的形状
            shape_x = curr_x.shape
            # 将当前级别数据重新整形以便进行卷积操作
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            # 应用小波卷积层和缩放模块
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            # 重新调整张量形状
            curr_x_tag = curr_x_tag.reshape(shape_x)

            # 分别存储低频和高频子带
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        # 初始化next_x_ll变量
        next_x_ll = 0

        # 反向遍历所有级别以执行逆小波变换
        for i in range(self.wt_levels - 1, -1, -1):
            # 从列表中取出对应级别的低频和高频子带
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            # 从列表中取出对应级别的原始形状
            curr_shape = shapes_in_levels.pop()

            # 更新当前级别的低频子带
            curr_x_ll = curr_x_ll + next_x_ll

            # 将低频和高频子带组合在一起
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            # 执行逆小波变换
            next_x_ll = self.iwt_function(curr_x)

            # 裁剪结果以匹配原始形状
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        # 最终结果赋值给x_tag
        x_tag = next_x_ll
        # 确认所有的低频子带已经被正确处理
        assert len(x_ll_in_levels) == 0

        # 应用基础卷积层和缩放模块
        x = self.base_scale(self.base_conv(x))
        # 将基础卷积结果与小波变换结果相加
        x = x + x_tag

        # 如果设置了步长，则执行步长操作
        if self.do_stride is not None:
            x = self.do_stride(x)

        # 返回最终结果
        return x

if __name__ == '__main__':
    # 创建一个WTConv2d实例
    block = WTConv2d(32, 32)
    # 创建随机输入张量
    input = torch.rand(1, 32, 64, 64)
    # 通过网络传递输入张量
    output = block(input)
    print("input.shape:", input.shape)
    print("output.shape:", output.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")