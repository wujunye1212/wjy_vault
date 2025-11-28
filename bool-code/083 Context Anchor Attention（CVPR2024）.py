from typing import Optional
import torch.nn as nn
import torch

"""
    论文地址：https://arxiv.org/pdf/2403.06258
    论文题目：Poly Kernel Inception Network for Remote Sensing Detection (CVPR 2024)
    中文题目：遥感检测的多核卷积网络(CVPR 2024)
    讲解视频：https://www.bilibili.com/video/BV18JkQYgEim/
        上下文锚点注意力机制(Context Anchor Attention ,CAA)：
            理论支撑：平均池化和1×1卷积获取局部区域特征、采用深度可分离条带卷积近似大核深度卷积
            模块功能：增加感受野，生成注意力权重，增强输出特征。
"""
class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,  # 输入通道数
            out_channels: int,  # 输出通道数
            kernel_size: int,  # 卷积核大小
            stride: int = 1,  # 步长，默认为1
            padding: int = 0,  # 填充，默认为0
            groups: int = 1,  # 组数，默认为1
            norm_cfg: Optional[dict] = None,  # 归一化配置，默认为None
            act_cfg: Optional[dict] = None):  # 激活函数配置，默认为None
        super().__init__()
        layers = []
        # 卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
        # 归一化层
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        # 激活层
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        # 组合所有层
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)  # 前向传播

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        # 如果需要，可以添加更多归一化类型
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        # 如果需要，可以添加更多激活函数类型
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

class CAA(nn.Module):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,  # 通道数
            h_kernel_size: int = 11,  # 水平卷积核大小，默认为11
            v_kernel_size: int = 11,  # 垂直卷积核大小，默认为11
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),  # 归一化配置
            act_cfg: Optional[dict] = dict(type='SiLU')):  # 激活函数配置
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)  # 平均池化层

        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)  # 卷积模块1

        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)  # 水平卷积模块

        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)  # 垂直卷积模块

        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)  # 卷积模块2

        self.act = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):
        avg = self.conv1(self.avg_pool(x))

        # 水平卷积模块 + 垂直卷积模块
        h_conv = self.h_conv(avg)
        v_conv = self.v_conv(h_conv)

        attn_factor = self.act(self.conv2(v_conv))
        return attn_factor

if __name__ == "__main__":

    input_tensor = torch.randn(1, 64, 128, 128)  # 随机生成输入张量
    caa = CAA(64)  # 创建CAA模块实例
    output_tensor = caa(input_tensor)  # 计算输出
    print("Input shape:", input_tensor.shape)  # 打印输入形状
    print("Output shape:", output_tensor.shape)  # 打印输出形状

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
