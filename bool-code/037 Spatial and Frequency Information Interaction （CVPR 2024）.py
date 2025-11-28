import torch
import torch.nn as nn

'''
    论文地址：https://openaccess.thecvf.com/content/CVPR2024/html/Cong_A_Semi-supervised_Nighttime_Dehazing_Baseline_with_Spatial-Frequency_Aware_and_Realistic_CVPR_2024_paper.html
    论文题目：A Semi-supervised Nighttime Dehazing Baseline with Spatial-Frequency Aware and Realistic Brightness Constraint（CVPR 2024）
    中文题目：具有空间频率感知和现实亮度约束的半监督夜间除雾基线模型
    讲解视频：https://www.bilibili.com/video/BV1pySxYkEi1/
      本地感知的双向非线性映射（Bidomain Local Perception and Nonlinear Mapping，BLPNM）：
                    Bidomain 非线性映射（BNM）是一种用于计算窗口注意力的方法，但计算窗口注意力不具有非线性表示能力。
                    因此，对频率域信息和空间域信息实现非线性映射，并在此基础上进行全局感知的自注意力计算。
'''
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:  # 如果同时使用偏置和归一化，则不使用偏置
            bias = False
        padding = kernel_size // 2  # 根据卷积核大小设置填充
        layers = list()  # 创建一个层列表
        if transpose:  # 如果是转置卷积
            padding = kernel_size // 2 - 1  # 转置卷积时调整填充
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))  # 添加转置卷积层
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))  # 添加普通卷积层
        if norm:  # 如果使用批归一化
            layers.append(nn.BatchNorm2d(out_channel))  # 添加批归一化层
        if relu:  # 如果使用激活函数
            layers.append(nn.GELU())  # 添加GELU激活函数
        self.main = nn.Sequential(*layers)  # 将所有层组成序列

    # 前向传播函数
    def forward(self, x):
        return self.main(x)  # 应用前面定义的所有层

# 定义空间块
class SpaBlock(nn.Module):
    # 初始化函数
    def __init__(self, nc):
        super(SpaBlock, self).__init__()  # 调用父类初始化方法
        in_channel = nc
        out_channel = nc
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)  # 第一层卷积
        self.trans_layer = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)  # 过渡层
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)  # 第二层卷积
    def forward(self, x):
        out = self.conv1(x)  # 经过第一层卷积
        out = self.trans_layer(out)  # 经过过渡层
        out = self.conv2(out)  # 经过第二层卷积
        return out + x  # 加上原始输入形成残差连接

# 定义SE（Squeeze-and-Excitation）层
class SELayer(nn.Module):
    # 初始化函数
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()  # 调用父类初始化方法
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 线性变换降维
            nn.ReLU(inplace=True),  # ReLU激活
            nn.Linear(channel // reduction, channel, bias=False),  # 线性变换升维
            nn.Sigmoid()  # Sigmoid激活
        )

    # 前向传播函数
    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入尺寸
        y = self.avg_pool(x).view(b, c)  # 全局平均池化后展平
        y = self.fc(y).view(b, c, 1, 1)  # 经过线性变换并恢复形状
        return x * y.expand_as(x)  # 对输入进行加权

# 定义频谱动态聚合模块
# 代码讲解视频：https://www.bilibili.com/video/BV1CE1uYDEf4/
class Frequency_Spectrum_Dynamic_Aggregation(nn.Module):
    # 初始化函数
    def __init__(self, nc):
        super(Frequency_Spectrum_Dynamic_Aggregation, self).__init__()  # 调用父类初始化方法
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),  # 1x1卷积处理幅度
            nn.LeakyReLU(0.1, inplace=True),  # LeakyReLU激活
            SELayer(channel=nc),  # SE层
            nn.Conv2d(nc, nc, 1, 1, 0)  # 另一个1x1卷积
        )
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),  # 1x1卷积处理相位
            nn.LeakyReLU(0.1, inplace=True),  # LeakyReLU激活
            SELayer(channel=nc),  # SE层
            nn.Conv2d(nc, nc, 1, 1, 0)  # 另一个1x1卷积
        )

    # 前向传播函数
    def forward(self, x):
        _, _, H, W = x.shape  # 获取输入尺寸

        x_freq = torch.fft.rfft2(x, norm='backward')  # 计算二维实数FFT

        ori_mag = torch.abs(x_freq)  # 计算幅度
        mag = self.processmag(ori_mag)  # 处理幅度
        mag = ori_mag + mag  # 残差连接

        ori_pha = torch.angle(x_freq)  # 计算相位
        pha = self.processpha(ori_pha)  # 处理相位
        pha = ori_pha + pha  # 残差连接

        real = mag * torch.cos(pha)  # 计算实部
        imag = mag * torch.sin(pha)  # 计算虚部

        x_out = torch.complex(real, imag)  # 合成复数输出

        x_freq_spatial = torch.fft.irfft2(x_out, s=(H, W), norm='backward')  # 逆FFT转换回空间域
        return x_freq_spatial  # 返回结果

# 定义双域非线性映射模块
class BidomainNonlinearMapping(nn.Module):

    def __init__(self, in_nc):
        super(BidomainNonlinearMapping, self).__init__()
        self.spatial_process = SpaBlock(in_nc)  # 空间处理块
        self.frequency_process = Frequency_Spectrum_Dynamic_Aggregation(in_nc)  # 频率处理块
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 1, 1, 0)  # 用于合并两个域信息的1x1卷积

    # 前向传播函数
    def forward(self, x):
        _, _, H, W = x.shape  # 获取输入尺寸

        x_freq = self.frequency_process(x)  # 频率域处理  torch.Size([4, 32, 64, 64])
        x = self.spatial_process(x)  # 空间域处理    torch.Size([4, 32, 64, 64])

        xcat = torch.cat([x, x_freq], 1)  # 在通道维度上合并
        x_out = self.cat(xcat)  # 应用1x1卷积
        return x_out  # 返回最终输出

# 主程序入口
if __name__ == '__main__':
    block = BidomainNonlinearMapping(32)  # 实例化模型
    input = torch.rand(4, 32, 64, 64)  # 创建随机输入张量
    output = block(input)  # 计算模型输出
    print(input.size())  # 打印输入张量尺寸
    print(output.size())  # 打印输出张量尺寸