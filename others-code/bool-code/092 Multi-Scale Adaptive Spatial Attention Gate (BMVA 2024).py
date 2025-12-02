import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    论文地址：https://arxiv.org/abs/2407.21640
    论文题目：MSA2Net: Multi-scale Adaptive Attention-guided Network for Medical Image Segmentation (BMVA 2024)
    中文题目：MSA2Net：用于医学图像分割的多尺度自适应注意力引导网络
    讲解视频：https://www.bilibili.com/video/BV1266oYVENV/
        多尺度自适应空间注意力门控（Multi-Scale Adaptive Spatial Attention Gate ,MASAG）
            多尺度特征融合：整合局部上下文提取（通过深度可分离卷积和空洞卷积扩展编码器高分辨率空间细节 X）和全局上下文提取（通过通道池化捕获解码器语义信息 G），形成综合特征图。
            空间选择：将融合特征图投影到两个通道，计算空间选择性权重，得到强调关键区域的 X' 和 G'，并通过残差连接改善梯度流和特征利用。
            空间交互与交叉调制：通过空间权重将 X' 与 G' 的局部和全局上下文信息相互增强，融合得到包含详细和全局上下文的 U'。
            重新校准：对 U' 进行卷积和激活操作，生成注意力图，用于重新校准编码器输入 X，使其具备精确上下文特征后融入解码器。
"""
class GlobalExtraction(nn.Module):
    def __init__(self,dim = None):
        super().__init__()
        self.avgpool = self.globalavgchannelpool
        self.maxpool = self.globalmaxchannelpool
        self.proj = nn.Sequential(
            nn.Conv2d(2, 1, 1,1),
            nn.BatchNorm2d(1)
        )

    def globalavgchannelpool(self, x):
        x = x.mean(1, keepdim = True)  # 计算通道维度的平均值
        return x

    def globalmaxchannelpool(self, x):
        x = x.max(dim = 1, keepdim=True)[0]  # 计算通道维度的最大值
        return x

    def forward(self, x):
        x_ = x.clone()
        x = self.avgpool(x)    # 通过平均池化提取全局特征
        x2 = self.maxpool(x_)  # 通过最大池化提取全局特征
        cat = torch.cat((x,x2), dim = 1)  # 连接两种池化特征
        proj = self.proj(cat)  # 投影融合特征
        return proj

class ContextExtraction(nn.Module):
    def __init__(self, dim, reduction = None):
        super().__init__()
        self.reduction = 1 if reduction == None else 2
        self.dconv = self.DepthWiseConv2dx2(dim)
        self.proj = self.Proj(dim)

    def DepthWiseConv2dx2(self, dim):
        dconv = nn.Sequential(
            nn.Conv2d(in_channels = dim,
                  out_channels = dim,
                  kernel_size = 3,
                  padding = 1,
                  groups = dim),  # 第一次深度可分离卷积
            nn.BatchNorm2d(num_features = dim),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = dim,
                  out_channels = dim,
                  kernel_size = 3,
                  padding = 2,
                  dilation = 2),  # 第二次空洞卷积
            nn.BatchNorm2d(num_features = dim),
            nn.ReLU(inplace = True)
        )
        return dconv

    def Proj(self, dim):
        proj = nn.Sequential(
            nn.Conv2d(in_channels = dim,
                  out_channels = dim //self.reduction,
                  kernel_size = 1
                  ),  # 1x1卷积降维
            nn.BatchNorm2d(num_features = dim//self.reduction)
        )
        return proj

    def forward(self,x):
        x = self.dconv(x)  # 提取局部上下文特征
        x = self.proj(x)   # 特征降维
        return x

class MultiscaleFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local = ContextExtraction(dim)
        self.global_ = GlobalExtraction()
        self.bn = nn.BatchNorm2d(num_features=dim)

    def forward(self, x, g,):
        x = self.local(x)    # 提取局部上下文特征
        g = self.global_(g)  # 提取全局通道注意力特征
        fuse = self.bn(x + g)  # 特征融合并进行标准化
        return fuse

class MultiScaleGatedAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.multi = MultiscaleFusion(dim)
        self.selection = nn.Conv2d(dim, 2,1)
        self.proj = nn.Conv2d(dim, dim,1)
        self.bn = nn.BatchNorm2d(dim)
        self.bn_2 = nn.BatchNorm2d(dim)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim,
                      kernel_size=1, stride=1))

    def forward(self,x,g):
        x_ = x.clone()  # 保存输入特征的副本
        g_ = g.clone()  # 保存门控特征的副本

        # 第一阶段：多尺度特征提取与融合【粉色部分】
        multi = self.multi(x, g)  # 融合局部和全局特征

        # 第二阶段：自适应特征选择【黄色部分】
        multi = self.selection(multi)  # 生成特征选择权重
        attention_weights = F.softmax(multi, dim=1)  # 权重归一化
        A, B = attention_weights.split(1, dim=1)  # 分离两个特征通道的权重
        x_att = A.expand_as(x_) * x_  # 应用特征选择权重到输入特征
        g_att = B.expand_as(g_) * g_  # 应用特征选择权重到门控特征
        x_att = x_att + x_  # 残差连接
        g_att = g_att + g_  # 残差连接

        # 第三阶段：特征交互与增强【蓝色部分】
        x_sig = torch.sigmoid(x_att)  # 生成输入特征的门控信号
        g_att_2 = x_sig * g_att  # 输入特征调制门控特征

        g_sig = torch.sigmoid(g_att)  # 生成门控特征的门控信号
        x_att_2 = g_sig * x_att  # 门控特征调制输入特征

        interaction = x_att_2 * g_att_2  # 特征交互融合

        # 第四阶段：特征重校准【紫色部分】
        projected = torch.sigmoid(self.bn(self.proj(interaction)))  # 特征投影与归一化
        weighted = projected * x_  # 特征重校准
        y = self.conv_block(weighted)  # 最终特征提取
        y = self.bn_2(y)  # 输出特征归一化
        return y

if __name__ == '__main__':
    x1 = torch.randn(1, 64, 32,32)
    x2 = torch.randn(1, 64, 32,32)
    Model = MultiScaleGatedAttn(dim=64)
    out = Model(x1,x2)
    print(out.shape)
    print("Input size:", x1.size(), x2.shape)  # 打印输入张量尺寸
    print("Output size:", out.size())  # 打印输出张量尺寸
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")