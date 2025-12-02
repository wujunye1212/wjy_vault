import math
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
来自TGRS 2024  一区 小目标检测任务论文   
即插即用模块： EFC 增强层间特征相关性  （增强特征融合模块）
含二次创新模块  MSEF 多尺度有效融合模块  MSEF对EFC模块进行二次创新，效果优于EFC模块，可以直接拿去发小论文  冲sci一区或二区

本文摘要内容分解讲解：
在无人机影像中检测小目标是一项极具挑战性的任务，由于目标分辨率低且背景复杂，会导致特征信息有限。---任务背景，所做任务数据集特点

多尺度特征融合可以通过捕获不同尺度的信息来增强检测效果，但传统策略存在不足。
简单的特征拼接或加法操作未能充分利用多尺度融合的优势，导致特征间的相关性不足。
这种不足尤其在复杂背景和密集目标区域中，限制了小目标的检测能力。  ---之前存在的方法，效果略显不足.

为了应对这一问题并有效利用有限的计算资源，我们提出了一种基于增强层间特征相关性（EFC）的轻量级融合策略，
取代传统FPN中的特征融合策略。    ---引出本文创新点，EFC

由于特征金字塔中不同层的语义表达存在不一致性，EFC引入了分组特征聚焦单元（GFF），
通过关注不同特征的上下文信息，增强各层的特征相关性。
同时，多层特征重构模块（MFR）能够有效地重构和转换特征金字塔中强弱特征的信息，
减少冗余融合，并在深层网络中保留更多关于小目标的信息。
值得注意的是，该方法具有即插即用的特点，可广泛应用于各种基础网络。  --- 对本文EFC创新模块的简单介绍

在VisDrone、UAVDT和COCO数据集上的广泛实验与全面评估表明，该方法卓有成效。
能够有效提高检测精度，同时降低网络模型的参数量和计算量GFLOPs。 --- 通过做广泛实验，证明本文创新的有效性

EFC模块的作用：
EFC模块旨在解决传统特征融合策略的不足之处，特别是对于小目标检测的场景。
具体作用包括：
1.增强特征相关性：通过对多层特征的空间和语义信息进行更高效的关联和融合，提高小目标特征的表达能力。
2.减少冗余特征：在特征金字塔网络（FPN）的融合阶段，通过更精细的特征重构，减少冗余特征，保留更多对小目标检测有用的信息。
3.轻量化设计：通过优化融合模块的复杂度，减少计算资源消耗，同时提升小目标检测精度

EFC模块的原理：
EFC模块包含两个核心子模块：
第一个是：分组特征聚焦单元GFF模块
空间聚焦：通过生成空间权重，对特征图中的不同区域进行加权，提升关键区域的表达能力。
特征相关性增强：将特征图按通道分组，分别进行局部交互和融合，从而提升通道间的相关性。
空间映射归一化：对聚合后的特征进行标准化处理，增强其空间位置信息。

第二个是：多层特征重构MFR模块
特征分离：将强特征和弱特征分开处理，避免强特征被弱特征干扰，同时针对弱特征进行独立优化。
特征转换：对强特征进行精细处理（如1×1卷积），对弱特征采用轻量化方法（如深度可分离卷积）以提取更多信息。
逐层融合：将重构后的特征再次融合，生成更具语义表达力的特征图，特别针对小目标的细节和语义表达进行优化。

适用于：小目标检测任务，小目标分割任务， 图像增强任务，图像分类任务，暗光增强任务
       超分图像任务，遥感图像任务等所有计算机视觉CV任务通用的即插即用模块
'''
class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
class EFC(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(c2)
        self.sigomid = nn.Sigmoid()
        self.group_num = 16
        self.eps = 1e-10
        self.gamma = nn.Parameter(torch.randn(c2, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c2, 1, 1))
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c2, c2, 1, 1),
            nn.ReLU(True),
            nn.Softmax(dim=1),
        )
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2)
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.one = c2
        self.two = c2
        self.conv4_gobal = nn.Conv2d(c2, 1, kernel_size=1, stride=1)
        for group_id in range(0, 4):
            self.interact = nn.Conv2d(c2 // 4, c2 // 4, 1, 1, )

    def forward(self, x1, x2):

        global_conv1 = self.conv1(x1)
        bn_x = self.bn(global_conv1)
        weight_1 = self.sigomid(bn_x)
        global_conv2 = self.conv2(x2)
        bn_x2 = self.bn(global_conv2)
        weight_2 = self.sigomid(bn_x2)
        X_GOBAL = global_conv1 + global_conv2
        x_conv4 = self.conv4_gobal(X_GOBAL)
        X_4_sigmoid = self.sigomid(x_conv4)
        X_ = X_4_sigmoid * X_GOBAL
        X_ = X_.chunk(4, dim=1)
        out = []
        for group_id in range(0, 4):
            out_1 = self.interact(X_[group_id])
            N, C, H, W = out_1.size()
            x_1_map = out_1.reshape(N, 1, -1)
            mean_1 = x_1_map.mean(dim=2, keepdim=True)
            x_1_av = x_1_map / mean_1
            x_2_2 = F.softmax(x_1_av, dim=1)
            x1 = x_2_2.reshape(N, C, H, W)
            x1 = X_[group_id] * x1
            out.append(x1)
        out = torch.cat([out[0], out[1], out[2], out[3]], dim=1)
        N, C, H, W = out.size()
        x_add_1 = out.reshape(N, self.group_num, -1)
        N, C, H, W = X_GOBAL.size()
        x_shape_1 = X_GOBAL.reshape(N, self.group_num, -1)
        mean_1 = x_shape_1.mean(dim=2, keepdim=True)
        std_1 = x_shape_1.std(dim=2, keepdim=True)
        x_guiyi = (x_add_1 - mean_1) / (std_1 + self.eps)
        x_guiyi_1 = x_guiyi.reshape(N, C, H, W)
        x_gui = (x_guiyi_1 * self.gamma + self.beta)

        weight_x3 = self.Apt(X_GOBAL)
        reweights = self.sigomid(weight_x3)
        x_up_1 = reweights >= weight_1
        x_low_1 = reweights < weight_1
        x_up_2 = reweights >= weight_2
        x_low_2 = reweights < weight_2
        x_up = x_up_1 * X_GOBAL + x_up_2 * X_GOBAL
        x_low = x_low_1 * X_GOBAL + x_low_2 * X_GOBAL
        x11_up_dwc = self.dwconv(x_low)
        x11_up_dwc = self.conv3(x11_up_dwc)
        x_so = self.gate_genator(x_low)
        x11_up_dwc = x11_up_dwc * x_so
        x22_low_pw = self.conv4(x_up)
        xL = x11_up_dwc + x22_low_pw
        xL = xL + x_gui
        return xL

# 二次创新模块 MSEF 可以直接拿去发小论文，冲sci 一区或二区
'''
MSEF 多尺度有效融合模块

MSEF 多尺度有效融合模块的内容介绍：
MSEF 多尺度有效融合模块
MSEF模块通过改进特征金字塔中不同层次特征的相关性，使深层语义特征与浅层高分辨率特征更加协调。
增强的相关性提升了多尺度特征融合的表达能力，尤其是针对小目标检测或是其它计算机视觉任务。
这个模块包含以下四个核心子模块作用：
GFF子模块：通过提取上下文信息和特征聚焦，增强了小目标在特征图中的表示能力，减少了目标在复杂背景中被忽略的可能性。
MFR子模块：分离并重构强特征和弱特征，最大限度地保留了小目标的细节和语义信息，同时避免了特征丢失。
通道注意力子模块：它对每个通道进行全局平均池化，然后通过1D卷积来捕捉通道之间的交互信息。
这种方法避免了降维问题，确保模型能够有效地聚焦在最相关的通道特征上，增强重要通道特征。
位置注意力子模块：对特征图的水平和垂直轴进行位置注意力处理，通过池化操作获取空间结构信息。
这一步有助于更准确地定位小目标的空间位置，增强对关键空间区域的关注。

'''

class MSEF(nn.Module): #多尺度有效融合模块
    def __init__(self, c1, c2):
        super().__init__()
        self.channel_att = channel_att(c2)
        self.local_att = local_att(c2)

        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(c2)
        self.sigomid = nn.Sigmoid()
        self.group_num = 16
        self.eps = 1e-10
        self.gamma = nn.Parameter(torch.randn(c2, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c2, 1, 1))
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c2, c2, 1, 1),
            nn.ReLU(True),
            nn.Softmax(dim=1),
        )
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2)
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.one = c2
        self.two = c2
        self.conv4_gobal = nn.Conv2d(c2, 1, kernel_size=1, stride=1)
        for group_id in range(0, 4):
            self.interact = nn.Conv2d(c2 // 4, c2 // 4, 1, 1, )

    def forward(self, x1, x2):

        global_conv1 = self.conv1(x1)
        bn_x = self.bn(global_conv1)
        weight_1 = self.sigomid(bn_x)
        global_conv2 = self.conv2(x2)
        bn_x2 = self.bn(global_conv2)
        weight_2 = self.sigomid(bn_x2)
        X_GOBAL = global_conv1 + global_conv2

        temp = self.channel_att(X_GOBAL)

        x_conv4 = self.conv4_gobal(X_GOBAL)
        X_4_sigmoid = self.sigomid(x_conv4)
        X_ = X_4_sigmoid * X_GOBAL
        X_ = X_.chunk(4, dim=1)
        out = []
        for group_id in range(0, 4):
            out_1 = self.interact(X_[group_id])
            N, C, H, W = out_1.size()
            x_1_map = out_1.reshape(N, 1, -1)
            mean_1 = x_1_map.mean(dim=2, keepdim=True)
            x_1_av = x_1_map / mean_1
            x_2_2 = F.softmax(x_1_av, dim=1)
            x1 = x_2_2.reshape(N, C, H, W)
            x1 = X_[group_id] * x1
            out.append(x1)
        out = torch.cat([out[0], out[1], out[2], out[3]], dim=1)
        N, C, H, W = out.size()
        x_add_1 = out.reshape(N, self.group_num, -1)
        N, C, H, W = X_GOBAL.size()
        x_shape_1 = X_GOBAL.reshape(N, self.group_num, -1)
        mean_1 = x_shape_1.mean(dim=2, keepdim=True)
        std_1 = x_shape_1.std(dim=2, keepdim=True)
        x_guiyi = (x_add_1 - mean_1) / (std_1 + self.eps)
        x_guiyi_1 = x_guiyi.reshape(N, C, H, W)
        x_gui = (x_guiyi_1 * self.gamma + self.beta)

        weight_x3 = self.Apt(X_GOBAL)
        reweights = self.sigomid(weight_x3)
        x_up_1 = reweights >= weight_1
        x_low_1 = reweights < weight_1
        x_up_2 = reweights >= weight_2
        x_low_2 = reweights < weight_2
        x_up = x_up_1 * X_GOBAL + x_up_2 * X_GOBAL
        x_low = x_low_1 * X_GOBAL + x_low_2 * X_GOBAL
        x11_up_dwc = self.dwconv(x_low)
        x11_up_dwc = self.conv3(x11_up_dwc)
        x_so = self.gate_genator(x_low)
        x11_up_dwc = x11_up_dwc * x_so
        x22_low_pw = self.conv4(x_up)
        xL = x11_up_dwc + x22_low_pw

        xL = xL + x_gui + temp
        out = self.local_att(xL)
        return out


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input1 = torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 64, 64, 64)
    # 初始化EFC模块并设定通道维度
    EFC_module = EFC(c1=32,c2=64) #c1表示input1通道数，c2表示input2通道数，
    output =EFC_module(input1,input2)#进行前向传播，输出通道数是C2
    # 输出结果的形状
    print("EFC_输入张量的形状：", input2.shape)
    print("EFC_输出张量的形状：", output.shape)

    # 初始化MSEF模块并设定通道维度
    MSEF_module = MSEF(c1=32, c2=64)
    output = MSEF_module(input1, input2)  # 进行前向传播，输出通道数是C2
    # 输出结果的形状
    print("二次创新MSEF_输入张量的形状：", input2.shape)
    print("二次创新MSEF_输出张量的形状：", output.shape)
