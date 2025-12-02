import torch.nn as nn
import torch
'''
    论文地址：https://arxiv.org/pdf/2303.15446.pdf
    论文题目：MAGNet: Multi-scale Awareness and Global fusion Network for RGB-D salient object detection（KBS 2024）
    中文题目：MAGNet：多尺度感知和全局融合网络用于RGB-D显著对象检测
    讲解视频：https://www.bilibili.com/video/BV1vwSSYbESd/
        多尺度感知融合模块（Multi-scale Awareness Fusion Module，MAFM)：
        启发：在显著目标检测任务中，RGB图像包含丰富的细节信息。而深度图像包含了每个像素点到相机的距离信息，代表了图像中的物体距离，包含更多的空间信息。
        做法：首先，将RGB和深度特征图在通道维度拼接，然后经过深度可分离卷积层、归一化层、GELU后，再进行点卷积、BN层和GELU降低特征图维数。随后传入多头
             混合卷积模块（MHMC）获得广泛特征关联，经过主分支和两个辅助分支，再将上述三个分支的特征图逐元素相加，最后由GELU激活函数得到融合后的特征图。
'''

class COI(nn.Module):
    def __init__(self, inc, k=3, p=1):  # 初始化函数，设置输入通道数inc，卷积核大小k默认为3，填充p默认为1
        super().__init__()
        self.outc = inc  # 输出通道数等于输入通道数
        self.dw = nn.Conv2d(inc, self.outc, kernel_size=k, padding=p, groups=inc)  # 深度可分离卷积
        self.conv1_1 = nn.Conv2d(inc, self.outc, kernel_size=1, stride=1)  # 1x1卷积
        self.bn1 = nn.BatchNorm2d(self.outc)  # 第一个批归一化层
        self.bn2 = nn.BatchNorm2d(self.outc)  # 第二个批归一化层
        self.bn3 = nn.BatchNorm2d(self.outc)  # 第三个批归一化层
        self.act = nn.GELU()  # 激活函数

    def forward(self, x):
        shortcut = self.bn1(x)

        x_dw = self.bn2(self.dw(x))  # 输入经过深度可分离卷积、第二个批归一化层

        x_conv1_1 = self.bn3(self.conv1_1(x))  # 输入经过1x1卷积、第三个批归一化层

        return self.act(shortcut + x_dw + x_conv1_1)  # 将上述三个路径的结果相加并激活后返回

# 定义一个多头混合注意力机制层
class MHMC(nn.Module):
    def __init__(self, dim, ca_num_heads=4, qkv_bias=True, proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()  # 调用父类的初始化方法
        self.ca_attention = ca_attention  # 设置多头注意力的数量
        self.dim = dim  # 设置特征维度
        self.ca_num_heads = ca_num_heads  # 设置多头注意力的头数

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."  # 断言特征维度必须能被头数整除

        self.act = nn.GELU()  # 激活函数
        self.proj = nn.Linear(dim, dim)  # 投影层
        self.proj_drop = nn.Dropout(proj_drop)  # 投影后的dropout层

        self.split_groups = self.dim // ca_num_heads  # 计算每个头的特征维度

        self.v = nn.Linear(dim, dim, bias=qkv_bias)  # V向量线性变换
        self.s = nn.Linear(dim, dim, bias=qkv_bias)  # K向量线性变换
        for i in range(self.ca_num_heads):  # 循环创建不同大小的深度可分离卷积
            local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
                                   padding=(1 + i), stride=1,
                                   groups=dim // self.ca_num_heads)  # 不同情境下的深度可分离卷积
            setattr(self, f"local_conv_{i + 1}", local_conv)  # 动态添加卷积层到实例中
        self.proj0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                               groups=self.split_groups)  # 1x1卷积扩展特征维度
        self.bn = nn.BatchNorm2d(dim * expand_ratio)  # 扩展后的特征批归一化
        self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)  # 1x1卷积降维回原特征维度

    def forward(self, x, H, W):  # 前向传播函数
        B, N, C = x.shape  # 获取输入张量的形状信息
        v = self.v(x)  # 应用值向量线性变换

        s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1, 2)  # 应用键向量线性变换，并调整形状
        for i in range(self.ca_num_heads):  # 对每个头应用对应的深度可分离卷积
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = s[i]  # 获取当前头的键向量
            s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)  # 应用卷积并调整形状
            if i == 0:
                s_out = s_i  # 如果是第一个头，则直接赋值
            else:
                s_out = torch.cat([s_out, s_i], 2)  # 否则将结果拼接起来
        s_out = s_out.reshape(B, C, H, W)  # 调整输出形状
        s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))  # 应用1x1卷积扩展-激活-批归一化-1x1卷积降维
        s_out = s_out.reshape(B, C, N).permute(0, 2, 1)  # 调整形状

        x = s_out * v

        x = self.proj(x)  # 应用投影层
        x = self.proj_drop(x)  # 应用dropout
        return x  # 返回处理后的特征

# 定义多尺度感知融合模块
class MAFM(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.outc = inc

        self.pre_att = nn.Sequential(  # 预注意力处理序列
            nn.Conv2d(inc * 2, inc * 2, kernel_size=3, padding=1, groups=inc * 2),  # 深度可分离卷积
            nn.BatchNorm2d(inc * 2),  # 批归一化
            nn.GELU(),  # 激活函数
            nn.Conv2d(inc * 2, inc, kernel_size=1),  # 1x1卷积降维
            nn.BatchNorm2d(inc),  # 批归一化
            nn.GELU()  # 激活函数
        )

        self.attention = MHMC(dim=inc)  # 注意力机制

        self.coi = COI(inc)  # 通道优化层

        self.pw = nn.Sequential(  # 逐点卷积序列
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1, stride=1),  # 1x1卷积
            nn.BatchNorm2d(inc),  # 批归一化
            nn.GELU()  # 激活函数
        )

    def forward(self, x, d):
        B, C, H, W = x.shape  # 获取输入张量的形状信息
        x_cat = torch.cat((x, d), dim=1)  # 拼接输入张量和深度图 torch.Size([1, 128, 32, 32])

        # DW-BN-GELU
        x_pre = self.pre_att(x_cat)  # 应用预注意力处理 torch.Size([1, 64, 32, 32])

        # MHMC ===> 特征关联
        x_reshape = x_pre.flatten(2).permute(0, 2, 1)  # 调整形状
        attention = self.attention(x_reshape, H, W)  # 应用注意力机制  torch.Size([1, 1024, 64])
        attention = attention.permute(0, 2, 1).reshape(B, C, H, W)  # 调整输出形状    torch.Size([1, 64, 32, 32])

        # 三个
        x_conv = self.coi(attention)  # 应用通道优化层 torch.Size([1, 64, 32, 32])

        # 加强版GELU
        x_conv = self.pw(x_conv)  # 应用逐点卷积  torch.Size([1, 64, 32, 32])

        return x_conv  # 返回处理后的特征

# 主程序入口
if __name__ == '__main__':
    mafm = MAFM(inc=64)  # 创建MAFM实例

    # 创建示例输入数据
    x = torch.randn(1, 64, 32, 32)  # 创建随机输入张量
    d = torch.randn(1, 64, 32, 32)  # 创建与输入张量相同形状的深度图

    output = mafm(x, d)  # 调用前向传播函数获取输出

    # 打印输入和输出的形状
    print(f"Input x shape: {x.shape}")  # 打印输入张量的形状
    print(f"Input d shape: {d.shape}")  # 打印深度图的形状
    print(f"Output shape: {output.shape}")  # 打印输出张量的形状

    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息