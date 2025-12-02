import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
from timm.models.layers import trunc_normal_, DropPath, to_2tuple  # 从timm库中导入一些常用的层函数
act_layer = nn.ReLU  # 定义激活函数为ReLU
ls_init_value = 1e-6  # 定义初始化参数值

# 多尺度特征对于密集预测任务（例如检测、分割）至关重要。主流方法通常利用主干提取多尺度特征，然后使用轻量级模块融合特征。
# 然而，这些方法将大部分计算资源分配给主干，因此这些方法中的多尺度特征融合被延迟，这可能导致特征融合不充分。
# 虽然有些方法从早期阶段开始执行特征融合，但它们要么未能充分利用高级特征来指导低级特征学习，要么结构复杂，导致性能不佳。

'''
论文地址：https://arxiv.org/abs/2302.06052 
论文题目：CEDNET: A CASCADE ENCODER-DECODER NETWORK FOR DENSE PREDICTION (ICLR 2024)
中文题目：CEDNet：用于密集预测的级联编码器-解码器网络
讲解视频：https://www.bilibili.com/video/BV1MEfXYmEuE/
    级联编码器 - 解码器模块（Cascade Encoder-Decoder,CED）
    该块包括用于空间特征交互的 token 混合器和用于通道特征交互的两层多层感知器 (MLP)。
    本文使用轻量级 7×7 深度卷积作为默认 token 混合器。作者提醒：更强大的 token 混合器可能会提高性能。
'''

class CED(nn.Module):  # 定义CED模块类

    def __init__(self, dim, drop_path=0., **kwargs):
        super().__init__()  # 调用父类构造函数

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # 深度可分离卷积，用于空间特征交互
        self.norm = nn.BatchNorm2d(dim)  # 归一化层，这里使用批标准化
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 第一个逐点卷积，使用线性层实现
        self.act = act_layer()  # 应用激活函数
        self.pwconv2 = nn.Linear(4 * dim, dim)  # 第二个逐点卷积

        self.gamma = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None  # 可学习的缩放参数

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # 随机丢弃路径，防止过拟合

    def forward(self, x):
        input = x  # 保存输入以供残差连接

        x = self.dwconv(x)  # 应用深度可分离卷积
        x = self.norm(x)  # 批标准化

        # 因为全连接层默认对最后一个维度进行变化，因此需要调整维度
        ##### 这里可以对MLPs进行改进，轻量化方向~
        ##### https://www.bilibili.com/video/BV1ENs9eiE2Z/
        x = x.permute(0, 2, 3, 1)  # 改变维度顺序以适应全连接层 (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)  # 应用第一个逐点卷积
        x = self.act(x)  # 激活
        x = self.pwconv2(x)  # 应用第二个逐点卷积
        if self.gamma is not None:
            x = self.gamma * x  # 如果定义了gamma，则进行缩放
        x = x.permute(0, 3, 1, 2)  # 将维度顺序改回 (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)  # 添加残差连接，并应用随机丢弃路径

        return x  # 返回处理后的张量

if __name__ == "__main__":
    input = torch.randn(1, 64, 32, 32)  # 创建随机输入张量
    model = CED(64)  # 实例化CED模型
    output = model(input)  # 前向传播
    print('input_size:', input.size())  # 打印输入尺寸
    print('output_size:', output.size())  # 打印输出尺寸

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
