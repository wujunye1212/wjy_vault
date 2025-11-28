import torch
import torch.nn as nn
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

"""
论文地址：https://openaccess.thecvf.com/content/CVPR2024/papers/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.pdf
论文题目：EMCAD: Efficient Multi-scale Convolutional Attention Decoding for MedicalImage Segmentation（CVPR2024）
讲解视频：https://www.bilibili.com/video/BV16UFWetEfE/
"""
'''
    Large-kernel grouped attention gate (LGAG) 用于逐步结合特征图与注意力系数，激活高相关性特征，并抑制不相关的特征，
高层特征的门控信号来控制网络不同阶段间的信息流动，能够有效地融合来自skip连接的信息，以更少的计算在更大的局部上下文中捕获显著特征。

    在LGAG机制中，使用分别的3x3组卷积GCg和GCx来处理g和x。这些卷积特征然后通过批量归一化BN进行归一化，并通过元素级加法合并。
    结果特征图通过ReLU层激活之后，应用一个1x1卷积后接BN层以获得单通道特征图。
    最后，将该单通道特征图通过Sigmoid激活函数以产生注意力权重。

'''
# 其他类型的层可以定义在这里（例如，nn.Linear等）
def _init_weights(module, name, scheme=''):  # 定义一个初始化网络层权重的函数
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):  # 如果层是2D或3D的卷积层
        if scheme == 'normal':  # 正态分布初始化
            nn.init.normal_(module.weight, std=.02)  # 权重初始化为均值为0，标准差为0.02的正态分布
            if module.bias is not None:  # 如果有偏置项，则初始化为0
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':  # 截断正态分布初始化
            trunc_normal_tf_(module.weight, std=.02)  # 权重初始化为截断正态分布
            if module.bias is not None:  # 如果有偏置项，则初始化为0
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':  # Xavier初始化
            nn.init.xavier_normal_(module.weight)  # 权重根据Xavier方法初始化
            if module.bias is not None:  # 如果有偏置项，则初始化为0
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':  # Kaiming初始化
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')  # 权重根据Kaiming方法初始化
            if module.bias is not None:  # 如果有偏置项，则初始化为0
                nn.init.zeros_(module.bias)
        else:  # 默认初始化方式，类似efficientnet
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels  # 计算输出的fan out
            fan_out //= module.groups  # 根据groups调整fan out
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))  # 权重初始化为均值为0，方差为2/fan_out的正态分布
            if module.bias is not None:  # 如果有偏置项，则初始化为0
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):  # 如果层是2D或3D的批标准化层
        nn.init.constant_(module.weight, 1)  # 权重初始化为1
        nn.init.constant_(module.bias, 0)  # 偏置初始化为0
    elif isinstance(module, nn.LayerNorm):  # 如果层是层规范化层
        nn.init.constant_(module.weight, 1)  # 权重初始化为1
        nn.init.constant_(module.bias, 0)  # 偏置初始化为0

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):  # 定义一个创建激活层的函数
    act = act.lower()  # 将激活函数名称转换为小写
    if act == 'relu':  # 如果是ReLU激活函数
        layer = nn.ReLU(inplace)  # 创建ReLU层
    elif act == 'relu6':  # 如果是ReLU6激活函数
        layer = nn.ReLU6(inplace)  # 创建ReLU6层
    elif act == 'leakyrelu':  # 如果是LeakyReLU激活函数
        layer = nn.LeakyReLU(neg_slope, inplace)  # 创建LeakyReLU层
    elif act == 'prelu':  # 如果是PReLU激活函数
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)  # 创建PReLU层
    elif act == 'gelu':  # 如果是GELU激活函数
        layer = nn.GELU()  # 创建GELU层
    elif act == 'hswish':  # 如果是Hardswish激活函数
        layer = nn.Hardswish(inplace)  # 创建Hardswish层
    else:  # 如果激活函数类型未知，则抛出异常
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer  # 返回创建好的激活层实例

class LGAG(nn.Module):  # 定义一个继承自torch.nn.Module的类LGAG
    """
        这个类定义了一个大核组注意力门控（LGAG），它接收两个输入张量g（来自skip连接的特征）和x（上采样的特征），并对它们进行处理以生成注意力权重。
        这些权重随后可以用来调整特征图，以增强相关特征并抑制无关特征。这个机制有助于提高模型在特定任务上的表现，例如医学图像分割。
    """
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):  # 初始化函数
        super(LGAG, self).__init__()  # 调用父类构造函数
        if kernel_size == 1:  # 如果卷积核大小为1x1，则使用单一组卷积
            groups = 1
        self.W_g = nn.Sequential(  # 定义处理门控信号g的序列
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,bias=True),  # 使用给定的参数设置卷积操作
            nn.BatchNorm2d(F_int)  # 批量归一化层
        )
        self.W_x = nn.Sequential(  # 定义处理输入特征x的序列
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,bias=True),  # 使用给定的参数设置卷积操作
            nn.BatchNorm2d(F_int)  # 批量归一化层
        )
        self.psi = nn.Sequential(  # 定义生成注意力权重的序列
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),  # 1x1卷积层
            nn.BatchNorm2d(1),  # 批量归一化层
            nn.Sigmoid()  # Sigmoid激活函数，用于生成0到1之间的注意力权重
        )
        self.activation = act_layer(activation, inplace=True)  # 根据提供的激活函数类型创建激活层实例

        self.init_weights('normal')  # 初始化权重

    def init_weights(self, scheme=''):  # 初始化权重的函数
        named_apply(partial(_init_weights, scheme=scheme), self)  # 使用partial函数封装_init_weights并应用到self的所有子模块

    def forward(self, g, x):  # 前向传播函数
        g1 = self.W_g(g)  # 处理门控信号g
        x1 = self.W_x(x)  # 处理输入特征x
        psi = self.activation(g1 + x1)  # 激活函数应用于g1和x1的和
        psi = self.psi(psi)  # 生成最终的注意力权重

        return x * psi

if __name__ == '__main__':

    LGAG = LGAG(F_g=32, F_l=32,F_int=64)
    x = torch.randn(1,32, 64,64)
    y = torch.randn(1,32, 64,64)

    output = LGAG(x,y)

    print(f"Output size: {output.size()}")

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
