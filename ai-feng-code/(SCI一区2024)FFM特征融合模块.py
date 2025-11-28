import torch
import torch.nn as nn

'''
来自SCI一区 2024 顶刊论文    该模块核心思想：充分利用双编码器之间互补优势
即插即用模块：FFM 特征融合模块   

FFM模块的主要作用是融合由CNN编码器和Transformer编码器生成的中间特征。
通过融合这些特征，FFM能够充分利用CNN在局部特征提取方面的优势以
及Transformer在全局语义信息捕捉方面的能力，从而帮助网络更好地获得更准确的特征。

FFM模块的工作原理主要包括以下3点：
1.通道注意力（ CA）：首先，通过通道注意力机制调整不同通道之间的权重，以强化重要特征并抑制不重要特征。
                   这一步骤有助于网络更加关注于对分割任务有用的特征。

2.跨域融合块（ CFB）：接着，利用跨域融合块进行特征融合。CFB能够处理来自不同编码器的特征，
                   通过一定的融合策略（如相加、相乘或拼接等）将它们结合起来。
                   这一步骤促进了CNN和Transformer特征之间的深度交互。

3.特征融合：进一步进行相关性增强及特征融合操作，以加强融合后的特征之间的关联性。
          这有助于网络更好地理解图像中的上下文信息，从而提高分割的准确性。

通俗的理解：FFM模块通过融合来自不同编码器的特征，并强化重要特征、促进特征之间的交互和关联性增强，
        从而提高了网络在图像分割任务中的性能。这一模块的设计体现了论文中提出的双编码器之间互补优势的核心思想之一，
        即充分利用CNN和Transformer的互补优势来改进裂缝分割的效果。
这个特征融合模块适用于：图像分割，目标检测，语义分割，图像增强，图像分类等所有计算机视觉CV任务都需要的一种即插即用模块
'''


class DSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out


class IDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out


class FFM(nn.Module):
    def __init__(self, dim1):
        super().__init__()
        dim2 = dim1
        self.trans_c = nn.Conv2d(dim1, dim2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.li1 = nn.Linear(dim2, dim2)
        self.li2 = nn.Linear(dim2, dim2)

        self.qx = DSC(dim2, dim2)
        self.kx = DSC(dim2, dim2)
        self.vx = DSC(dim2, dim2)
        self.projx = DSC(dim2, dim2)

        self.qy = DSC(dim2, dim2)
        self.ky = DSC(dim2, dim2)
        self.vy = DSC(dim2, dim2)
        self.projy = DSC(dim2, dim2)

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
        x = self.trans_c(x)

        avg_x = self.avg(x).permute(0, 2, 3, 1)
        avg_y = self.avg(y).permute(0, 2, 3, 1)
        x_weight = self.li1(avg_x)
        y_weight = self.li2(avg_y)
        x = x.permute(0, 2, 3, 1) * x_weight
        y = y.permute(0, 2, 3, 1) * y_weight

        out1 = x * y
        out1 = out1.permute(0, 3, 1, 2)

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        qy = self.qy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        kx = self.kx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        vx = self.vx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)

        attnx = (qy @ kx.transpose(-2, -1)) * (C ** -0.5)
        attnx = attnx.softmax(dim=-1)
        attnx = (attnx @ vx).transpose(2, 3).reshape(B, H // 4, w // 4, 4, 4, C)
        attnx = attnx.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attnx = self.projx(attnx)

        qx = self.qx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        ky = self.ky(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        vy = self.vy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)

        attny = (qx @ ky.transpose(-2, -1)) * (C ** -0.5)
        attny = attny.softmax(dim=-1)
        attny = (attny @ vy).transpose(2, 3).reshape(B, H // 4, w // 4, 4, 4, C)
        attny = attny.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attny = self.projy(attny)
        out2 = torch.cat([attnx, attny], dim=1)
        out2 = self.concat(out2)
        out = torch.cat([x, y, out1, out2], dim=1)
        out = self.fusion(out)
        return out

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # 解释一下：原来FFM模块要求输入的两个特征维度分别是：3维的和4维的。
    # 我对FFM模块的前向传播过程，稍微修改了一点，此时支持输入两个4维的特征图
    input1 = torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 32, 64, 64)

    # 初始化 FFM 模块并设置输入通道维度和输出通道维度
    FFM_module = FFM(32)
    # 将输入张量传入 FFM 模块
    output = FFM_module(input1, input2)
    # 输出结果的形状
    print("FFM_输入张量的形状：", input1.shape)
    print("FFM_输出张量的形状：", input2.shape)