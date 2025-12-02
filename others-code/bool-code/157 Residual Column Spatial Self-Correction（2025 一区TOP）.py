import torch.nn as nn
import torch
import torch.nn.functional as F

"""    
    论文地址：https://arxiv.org/pdf/2401.15578
    论文题目：ASCNet: Asymmetric Sampling Correction Network for Infrared Image Destriping （TGRS 2025）
    中文题目：ASCNet：用于红外图像去条带的非对称采样校正网络 （TGRS 2025）
    讲解视频：https://www.bilibili.com/video/BV1kbj7z5Ebt/
    残差列空间自校正（Residual Column Spatial Self-Correction，RCSSC）：
        实际意义：①列特征不均匀：条纹噪声会导致图像中同一列的像素响应不一致。
                ②空间结构混淆：难以区分条纹与真实边缘。③全局依赖缺失：条纹噪声可能在全局范围内呈现长程相关性。
        实现方式：列注意力 + 空间注意力 + 自校准。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

# 通道池化模块：将输入特征图在通道维度上进行最大池化和平均池化，并将结果拼接
class ChannelPool(nn.Module):
    def forward(self, x):
        # 在通道维度上进行最大池化和平均池化，然后在通道维度上拼接
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

# 基础卷积模块：包含卷积层、批归一化层和激活函数
class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        # 定义卷积层
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        # 定义批归一化层（可选）
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        # 定义LeakyReLU激活函数（可选）
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# 通道注意力层：通过学习通道间的依赖关系来增强特征表示
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        # 定义宽度方向的自适应平均池化和最大池化
        self.avgPoolW = nn.AdaptiveAvgPool2d((1, None))
        self.maxPoolW = nn.AdaptiveMaxPool2d((1, None))

        # 定义1x1卷积层、批归一化层和激活函数
        self.conv_1x1 = nn.Conv2d(in_channels=2 * channel, out_channels=2 * channel, kernel_size=1, padding=0, stride=1,
                                  bias=False)
        self.bn = nn.BatchNorm2d(2 * channel, eps=1e-5, momentum=0.01, affine=True)
        self.Relu = nn.LeakyReLU()

        # 定义高度方向的激发操作：通过降维和升维来学习通道间的关系
        self.F_h = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        # 定义宽度方向的激发操作
        self.F_w = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        # 定义Sigmoid激活函数，将输出值映射到[0,1]区间
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.size()  # 获取输入张量的尺寸
        res = x  # 保存输入作为残差连接
        # 对输入进行平均池化和最大池化，并在通道维度上拼接
        x_cat = torch.cat([self.avgPoolW(x), self.maxPoolW(x)], 1)
        # 通过卷积、批归一化和激活函数处理拼接后的特征
        x = self.Relu(self.bn(self.conv_1x1(x_cat)))
        # 将特征拆分为两部分，分别用于高度和宽度方向的注意力计算
        x_1, x_2 = x.split(C, 1)

        x_1 = self.F_h(x_1)  # 高度方向的通道注意力
        x_2 = self.F_w(x_2)  # 宽度方向的通道注意力
        s_h = self.sigmoid(x_1)  # 将高度方向的注意力值归一化
        s_w = self.sigmoid(x_2)  # 将宽度方向的注意力值归一化

        # 将原始输入与注意力权重相乘，增强重要特征
        out = res * s_h.expand_as(res) * s_w.expand_as(res)

        return out

# 空间注意力层：通过学习空间位置的依赖关系来增强特征表示
class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()  # 使用通道池化压缩通道维度
        # 定义空间注意力模块，将通道池化的结果映射到一个空间注意力图
        self.spatial = Basic(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bn=False, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)  # 压缩通道维度
        x_out = self.spatial(x_compress)  # 生成空间注意力图
        scale = torch.sigmoid(x_out)  # 将注意力图的值归一化到[0,1]
        return x * scale  # 将原始输入与注意力权重相乘


# 残差列空间自校正模块：结合空间注意力和通道注意力来增强特征表示
class RCSSC(nn.Module):
    def __init__(self, n_feat, reduction=16):
        super(RCSSC, self).__init__()
        pooling_r = 4  # 池化比例
        # 定义头部卷积层，提取初始特征
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(),
        )

        # 定义空间校正分支：通过池化和卷积来捕获全局信息
        self.SC = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(n_feat)
        )
        self.SA = spatial_attn_layer()
        self.CA = CALayer(n_feat, reduction)

        # 定义特征融合卷积层：将空间注意力和通道注意力的结果融合
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat, kernel_size=1),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True)
        )
        self.ReLU = nn.LeakyReLU()

        # 定义尾部卷积层：进一步处理特征并生成最终输出
        self.tail = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1)

    def forward(self, x):
        # 图(a)中的第二个3X3
        res = x  # 【最下方的跳跃连接】
        x = self.head(x)  # 图(a)中的第一个3X3

        # CSSC 开始===========
        sa_branch = self.SA(x)  # 空间注意力分支（SAB） 图(b)
        ca_branch = self.CA(x)  # 列注意力分支（CAB） 图(c)
        x1 = torch.cat([sa_branch, ca_branch], dim=1)  # 拼接两个注意力分支的结果
        x1 = self.conv1x1(x1)  # 通过卷积融合两个注意力分支的特征

        # 空间校正：将原始特征与经过池化和卷积处理后的特征相加并通过sigmoid归一化
        A = self.SC(x) # 自校准分支（SCB） 图(d)
        x2 = torch.sigmoid(torch.add(x, F.interpolate(A, x.size()[2:])))
        # CSSC 结束===========

        out = torch.mul(x1, x2)  # 将注意力融合后的特征与空间校正的结果相乘
        out = self.tail(out)  # 通过尾部卷积层处理结果 图(a)中的第二个3X3
        out = out + res  # 添加残差连接
        out = self.ReLU(out)  # 通过激活函数
        return out

if __name__ == '__main__':
    x = torch.randn(1, 32, 50, 50)
    model = RCSSC(n_feat=32)  # 创建RCSSC模块实例
    output = model(x)
    print(f"输入张量形状: {x.shape}")  # [1, 32, 50, 50]
    print(f"输出张量形状: {output.shape}")  # [1, 32, 25, 25]（空间维度下采样2倍）
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")