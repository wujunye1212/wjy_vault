import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
try:
    from mmcv.cnn import ConvModule, build_norm_layer
    from mmengine.model import BaseModule
    from mmengine.model import constant_init
    from mmengine.model.weight_init import trunc_normal_init, normal_init
except ImportError as e:
    pass
'''
来自CVPR2025顶会

即插即用模块: Mona 多认知视觉适配器模块 （全称：Multi-cognitive Visual Adapter）
两个二次创新模块: DyTMona, MoCAA 好好编故事，可以直接拿去冲SCI一区、二区、三区

本文核心内容：
预训练和微调可以提高视觉任务中的传输效率和性能。最近的增量调整方法为视觉分类任务提供了更多选择。
尽管他们取得了成功，但现有的视觉增量调整艺术未能超过在对象检测和分割等具有挑战性的任务上完全微调的上限。
为了找到完全微调的有竞争力的替代方案，我们提出了多认知视觉适配器 （Mona） 调整，这是一种新颖的基于适配器的调整方法。
首先，我们在适配器中引入了多个视觉友好型滤波器，以增强其处理视觉信号的能力，而以前的方法主要依赖于语言友好的线性耳罩。
其次，我们在适配器中添加缩放的归一化层，以调节虚拟滤波器的输入特征分布。
为了充分展示 Mona 的实用性和通用性，我们对多个表征视觉任务进行了实验，包括 COCO 上的实例分割、
ADE20K 上的语义分割、Pas cal VOC 上的目标检测、DOTA/STAR 上的定向对象检测和三个常见数据集上的年龄分类。
令人兴奋的结果表明，Mona 在所有这些任务上都超过了完全微调，并且是唯一一种在上述各种任务上优于完全微调的增量调优方法。
例如，与完全微调相比，Mona 在 COCO 数据集上实现了 1% 的性能提升。
综合结果表明，与完全微调相比，Mona 调优更适合保留和利用预训练模型的能力。

适用于：语义分割、目标检测、实例分割、图像分类、图像增强等等所有CV任务都用的上，通用的即插即用模块
'''
class CAA(nn.Module):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None ):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor
class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )
    def forward(self, x):
        AiFHG = x
        conv1_x = self.conv1_AiFHG(x)
        conv2_x = self.conv2_AiFHG(x)
        conv3_x = self.conv3_AiFHG(x)
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + AiFHG

        AiFHG = x

        x = self.projector(x)

        return AiFHG + x
class Mona(nn.Module):
    def __init__(self,in_dim,AiFHG=4):
        super().__init__()
        self.project1_AiFHG = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2_AiFHG = nn.Linear(64, in_dim)

        self.dropout_AiFHG = nn.Dropout(p=0.1)

        self.adapter_conv_AiFHG = MonaOp(64)

        self.norm_AiFHG = nn.LayerNorm(in_dim)
        self.gamma_AiFHG = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax_AiFHG = nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B, C, -1).transpose(-1, -2)
        AiFHG = x
        x = self.norm_AiFHG(x) * self.gamma_AiFHG + x * self.gammax_AiFHG
        project1 = self.project1_AiFHG(x)  #降维操作，减少计算量
        b, n, c = project1.shape
        h, w = H,W
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv_AiFHG(project1)  #使用多尺度卷积操作，3*3，5*5，7*7代表不同大小的卷积核
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)
        nonlinear = self.nonlinear(project1)   #使用激活函数
        nonlinear = self.dropout_AiFHG(nonlinear)
        project2 = self.project2_AiFHG(nonlinear)  #升维操作，还原方便残差连接
        out = AiFHG + project2
        out = out.reshape(B, H, W,C).permute(0,3,1,2)
        return out

#二次创新模块：MoCAA
class MoCAA(nn.Module):  #Multi-cognitive Context Anchor Attention
    def __init__(self,in_dim,AiFHG=4):
        super().__init__()
        self.project1_AiFHG = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2_AiFHG = nn.Linear(64, in_dim)

        self.dropout_AiFHG = nn.Dropout(p=0.1)

        self.adapter_conv_AiFHG = CAA(64)

        self.norm_AiFHG = nn.LayerNorm(in_dim)
        self.gamma_AiFHG = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax_AiFHG = nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B, C, -1).transpose(-1, -2)
        AiFHG = x
        x = self.norm_AiFHG(x) * self.gamma_AiFHG + x * self.gammax_AiFHG
        project1 = self.project1_AiFHG(x)
        b, n, c = project1.shape
        h, w = H,W
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv_AiFHG(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)
        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout_AiFHG(nonlinear)
        project2 = self.project2_AiFHG(nonlinear)
        out = AiFHG + project2
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=False, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x
#二次创新模块：DyTMona
class DyTMona(nn.Module):
    def __init__(self,in_dim,AiFHG=4):
        super().__init__()
        self.project1_AiFHG = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2_AiFHG = nn.Linear(64, in_dim)
        self.dropout_AiFHG = nn.Dropout(p=0.1)
        self.adapter_conv_AiFHG = MonaOp(64)
        self.norm_AiFHG = DynamicTanh(normalized_shape=in_dim)
        self.gamma_AiFHG = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax_AiFHG = nn.Parameter(torch.ones(in_dim))
    def forward(self, x):
        B,C,H,W = x.shape

        x_dyt = self.norm_AiFHG(x).reshape(B, C, -1).transpose(-1, -2)
        x = x.reshape(B, C, -1).transpose(-1, -2)
        x=x_dyt* self.gamma_AiFHG + x * self.gammax_AiFHG
        AiFHG = x
        project1 = self.project1_AiFHG(x)
        b, n, c = project1.shape
        h, w = H,W
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv_AiFHG(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)
        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout_AiFHG(nonlinear)
        project2 = self.project2_AiFHG(nonlinear)
        out = AiFHG + project2
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out
 #   输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    #创建Mona模块实例，32代表通道维度
    Mona = Mona(32)
    # 随机生成输入4维度张量：B, C, H, W
    input= torch.randn(1, 32,32,32)
    # 运行前向传递
    output = Mona(input)
    # 输出输入图片张量和输出图片张量的形状
    print("CV_Mona_input size:", input.size())
    print("CV_Mona_Output size:", output.size())

    #创建DyTMona模块实例，64代表通道维度
    DyTMona = DyTMona(64)
    # 随机生成输入4维度张量：B, C, H, W
    input= torch.randn(1, 64,32,32)
    # 运行前向传递
    output  = DyTMona(input)
    # 输出输入图片张量和输出图片张量的形状
    print("二次创新——DyTMona_input size:", input.size())
    print("二次创新——DyTMona_Output size:", output.size())

    #创建MCAA模块实例，128代表通道维度
    MoCAA = MoCAA(128)
    # 随机生成输入4维度张量：B, C, H, W
    input= torch.randn(1, 128,32,32)
    # 运行前向传递
    output = MoCAA(input)
    # 输出输入图片张量和输出图片张量的形状
    print("二次创新——MoCAA_input size:", input.size())
    print("二次创新——MoCAA_Output size:", output.size())
