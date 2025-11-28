import torch
import torch.nn as nn

"""
    论文地址：https://www.sciencedirect.com/science/article/pii/S0957417424018177
    论文题目：A dual encoder crack segmentation network with Haar wavelet-based high–low frequency attention (2024 一区TOP)
    中文题目：基于Haar小波高低频注意力的双编码器裂缝分割网络
    讲解视频：https://www.bilibili.com/video/BV1ibNbe5EC8/
        特征融合模块模块（Feature fusion module ，FFM）：
        解决问题：简单堆叠卷积会丢失细节信息。Transformer在自然语言处理和计算机视觉领域广泛应用，在裂缝分割中也有研究，但多基于现有骨干网络，
                                    未针对裂缝特性设计，且易受背景干扰。
        实现方式：先对CNN和Transformer编码器中间特征进行维度调整和通道注意力（CA）处理，
                再经跨域融合块（CFB）和矩阵乘法实现跨域融合与相关性增强，最后通过特征融合块（FFB）得到融合特征。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

# 定义深度可分离卷积类（Depthwise Separable Convolution）
class DSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        # 初始化输入通道数和输出通道数
        self.c_in = c_in
        self.c_out = c_out
        # 定义深度卷积层
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        # 定义逐点卷积层
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        # 先通过深度卷积，再通过逐点卷积
        out = self.dw(x)
        out = self.pw(out)
        return out


# 定义逆深度可分离卷积类（Inverse Depthwise Separable Convolution）
class IDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        # 初始化输入通道数和输出通道数
        self.c_in = c_in
        self.c_out = c_out
        # 定义深度卷积层
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)
        # 定义逐点卷积层
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        # 先通过逐点卷积，再通过深度卷积
        out = self.pw(x)
        out = self.dw(out)
        return out


# 定义特征融合模块（Feature Fusion Module）
class FFM(nn.Module):
    def __init__(self, dim1):
        super().__init__()
        dim2 = dim1
        # 定义一系列操作用于特征变换、池化、线性变换等
        self.trans_c = nn.Conv2d(dim1, dim2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.li1 = nn.Linear(dim2, dim2)
        self.li2 = nn.Linear(dim2, dim2)

        # 定义多个DSC实例用于处理x和y特征图
        self.qx = DSC(dim2, dim2)
        self.kx = DSC(dim2, dim2)
        self.vx = DSC(dim2, dim2)
        self.projx = DSC(dim2, dim2)

        self.qy = DSC(dim2, dim2)
        self.ky = DSC(dim2, dim2)
        self.vy = DSC(dim2, dim2)
        self.projy = DSC(dim2, dim2)

        # 定义连接层和融合序列
        self.concat = nn.Conv2d(dim2 * 2, dim2, 1)
        self.fusion = nn.Sequential(IDSC(dim2 * 4, dim2),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU(),
                                    DSC(dim2, dim2),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU(),
                                    nn.Conv2d(dim2, dim2, 1),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU())

    def forward(self, x, y):
        b, c, h, w = x.shape
        B, N, C = b, h * w, c
        H = W = h
        # 特征转换
        x = self.trans_c(x)

        # 平均池化并调整维度顺序[通道注意力]
        avg_x = self.avg(x).permute(0, 2, 3, 1)
        avg_y = self.avg(y).permute(0, 2, 3, 1)
        x_weight = self.li1(avg_x)
        y_weight = self.li2(avg_y)
        x = x.permute(0, 2, 3, 1) * x_weight
        y = y.permute(0, 2, 3, 1) * y_weight
        # 处理x和y特征图
        out1 = x * y
        out1 = out1.permute(0, 3, 1, 2)


        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        # 对y进行查询Q操作，对x进行键值 K V 对操作[排列1]
        qy = self.qy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8, 16, C // 8)
        kx = self.kx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8, 16, C // 8)
        vx = self.vx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8, 16, C // 8)
        attnx = (qy @ kx.transpose(-2, -1)) * (C ** -0.5)
        attnx = attnx.softmax(dim=-1)
        attnx = (attnx @ vx).transpose(2, 3).reshape(B, H // 4, w // 4, 4, 4, C)
        attnx = attnx.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attnx = self.projx(attnx)
        # 对x进行查询Q操作，对y进行键值K V对操作[排列2]
        qx = self.qx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8, 16, C // 8)
        ky = self.ky(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8, 16, C // 8)
        vy = self.vy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8, 16, C // 8)
        attny = (qx @ ky.transpose(-2, -1)) * (C ** -0.5)
        attny = attny.softmax(dim=-1)
        attny = (attny @ vy).transpose(2, 3).reshape(B, H // 4, w // 4, 4, 4, C)
        attny = attny.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attny = self.projy(attny)
        # 将[排列1]和[排列2]合并到一起
        out2 = torch.cat([attnx, attny], dim=1)
        out2 = self.concat(out2)

        # 将原始结果、CNN、Transfomer的特征结果合并到一起
        out = torch.cat([x, y, out1, out2], dim=1)
        out = self.fusion(out)
        return out

if __name__ == '__main__':
    input1 = torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 32, 64, 64)
    # 初始化FFM模块并设置输入通道维度和输出通道维度
    FFM_module = FFM(32)
    # 将输入张量传入FFM模块
    output = FFM_module(input1, input2)
    # 输出结果的形状
    print("输入张量的形状：", input1.shape)
    print("输出张量的形状：", output.shape)  # 修改了此处以打印正确的输出形状