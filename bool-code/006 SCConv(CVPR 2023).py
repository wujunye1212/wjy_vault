import torch
import torch.nn.functional as F
import torch.nn as nn

"""
论文地址：https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf
论文题目：SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy（CVPR 2023）
讲解视频：https://www.bilibili.com/video/BV1wFbAekEZG/
            卷积神经网络(CNN)在各种计算机视觉任务中取得了显著的性能，但这是以巨大的计算资源为代价的，部分原因是卷积层提取冗余特征。
            利用特征之间的空间和通道冗余来进行CNN压缩，并提出了一种高效的卷积模块，称为SCConv (spatial and channel reconstruction convolution)，
            以减少冗余计算并促进代表性特征的学习。
"""
# 定义一个二维组归一化层
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,  # 组的数量，默认为16
                 eps: float = 1e-10   # 为了数值稳定性添加到分母的小常数
                 ):
        super(GroupBatchnorm2d, self).__init__()  # 调用父类的构造函数
        assert c_num >= group_num  # 确保输入通道数量大于等于组的数量
        self.group_num = group_num  # 设置组的数量
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))  # 初始化缩放参数gamma
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))  # 初始化偏移参数beta
        self.eps = eps  # 设置小常数eps

    def forward(self, x):  # 前向传播方法
        N, C, H, W = x.size()  # 获取输入张量的尺寸
        x = x.view(N, self.group_num, -1)  # 将张量重新塑形，以便于按组计算均值和标准差  四维 - 三维

        mean = x.mean(dim=2, keepdim=True)  # 计算每组的均值
        std = x.std(dim=2, keepdim=True)  # 计算每组的标准差
        x = (x - mean) / (std + self.eps)  # 归一化

        x = x.view(N, C, H, W)  # 恢复原始形状 3维 - 4维
        return x * self.gamma + self.beta  # 应用缩放和平移

# Spatial Reconstruction Unit (SRU)
class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,  # 输出通道数量
                 group_num: int = 16,  # 组的数量
                 gate_treshold: float = 0.5  # 门限值
                 ):
        super().__init__()  # 调用父类的构造函数
        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)  # 创建GroupBatchnorm2d实例
        self.gate_treshold = gate_treshold  # 设置门限值
        self.sigomid = nn.Sigmoid()  # 初始化Sigmoid激活函数

    def forward(self, x):

        gn_x = self.gn(x)  # 组归一化 GN层的可训练参数γ衡量特征图中空间信息的不同，空间信息越是丰富，γ越大。

        w_gamma = self.gn.gamma / sum(self.gn.gamma)  # 计算权重化的gamma，反映不同特征图的重要性
        reweigts = self.sigomid(gn_x * w_gamma)  # 计算重加权后的值

        # 门控机制，获得信息量大和信息量较少的两个特征图
        info_mask = reweigts >= self.gate_treshold  # 信息掩码
        noninfo_mask = reweigts < self.gate_treshold  # 非信息掩码

        x_1 = info_mask * x  # 保留信息部分
        x_2 = noninfo_mask * x  # 保留非信息部分

        x = self.reconstruct(x_1, x_2)  # 重构输出

        return x

    def reconstruct(self, x_1, x_2):  # 重构方法
        # 交叉相乘与cat，获得最终的输出特征：能够更加有效地联合两个特征 并且 加强特征之间的交互
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)  # 分割x_1
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)  # 分割x_2
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)  # 重构并连接

# Channel Reconstruction Unit (CRU)
class CRU(nn.Module):
    '''
    alpha: 0<alpha<1  # alpha应该在0到1之间
    '''

    def __init__(self,
                 op_channel: int,  # 操作通道数量
                 alpha: float = 1 / 2,  # 分割比例
                 squeeze_radio: int = 2,  # 压缩率
                 group_size: int = 2,  # 组大小
                 group_kernel_size: int = 3  # 组卷积核大小
                 ):
        super().__init__()  # 调用父类的构造函数

        self.up_channel = up_channel = int(alpha * op_channel)  # 计算上半部分通道数量
        self.low_channel = low_channel = op_channel - up_channel  # 计算下半部分通道数量

        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)  # 上半部分压缩卷积
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)  # 下半部分压缩卷积

        # 上半部分
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)  # 组卷积
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)  # 点卷积

        # 下半部分
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio,
                              kernel_size=1,bias=False)  # 点卷积
        self.advavg = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化

    def forward(self, x):  # 前向传播方法

        # 分割
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)  # 分割输入张量

        up, low = self.squeeze1(up), self.squeeze2(low)  # 应用压缩卷积

        # 变换
        Y1 = self.GWC(up) + self.PWC1(up)  # 上半部分变换
        Y2 = torch.cat([self.PWC2(low), low], dim=1)  # 下半部分变换

        # 融合
        out = torch.cat([Y1, Y2], dim=1)

        out = F.softmax(self.advavg(out), dim=1) * out  # 使用softmax进行通道注意力机制

        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)  # 再次分割

        return out1 + out2  # 返回融合后的结果

# 定义ScConv模块
class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,  # 操作通道数量
                 group_num: int = 16,  # 组的数量
                 gate_treshold: float = 0.5,  # 门限值

                 alpha: float = 1 / 2,  # 分割比例
                 squeeze_radio: int = 2,  # 压缩率
                 group_size: int = 2,  # 组大小
                 group_kernel_size: int = 3  # 组卷积核大小
                 ):
        super().__init__()  # 调用父类的构造函数
        self.SRU = SRU(op_channel,  # 创建SRU实例
                       group_num=group_num,
                       gate_treshold=gate_treshold
                       )
        self.CRU = CRU(op_channel,  # 创建CRU实例
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size
                       )

    def forward(self, x):  # 前向传播方法
        x = self.SRU(x)  # 应用SRU
        x = self.CRU(x)  # 应用CRU
        return x  # 返回最终结果

if __name__ == '__main__':
    # 生成随机输入张量
    input = torch.randn(1, 32, 64, 64)  # 创建一个随机输入张量
    model = ScConv(32)  # 创建ScConv模型实例
    # 执行前向传播
    output = model(input)
    print('input_size:', input.size())  # 打印输入张量的尺寸
    print('output_size:', output.size())  # 打印输出张量的尺寸

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")