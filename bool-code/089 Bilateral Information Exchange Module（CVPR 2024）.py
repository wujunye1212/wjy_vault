import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init

"""
    论文地址：https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_Bilateral_Event_Mining_and_Complementary_for_Event_Stream_Super-Resolution_CVPR_2024_paper.pdf
    论文题目：Bilateral Event Mining and Complementary for Event Stream Super-Resolution  (CVPR 2024)
    中文题目：双边事件挖掘与互补用于事件流超分辨率 (CVPR 2024)
    讲解视频：https://www.bilibili.com/video/BV17C6tYUE1p/
        双边信息交换模块（Bilateral Information Exchange Module , BIEM）
             提出问题：由于正负事件在相应的空间位置及其邻近区域表现出高度相关性，促进两种类型事件之间全局结构信息的选择性整合，
                        并减轻噪声的潜在误导影响。
"""

def initialize_weights(net_l, scale=0.1):
    # 如果输入不是列表，则将其转换为列表
    if not isinstance(net_l, list):
        net_l = [net_l]
    # 遍历网络中的每个模块
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming初始化卷积层权重
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # 缩放权重
                if m.bias is not None:
                    m.bias.data.zero_()  # 将偏置设为零
            elif isinstance(m, nn.Linear):
                # 使用Kaiming初始化线性层权重
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()  # 将偏置设为零
            elif isinstance(m, nn.BatchNorm2d):
                # 初始化BatchNorm层
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)  # 计算均值
        var = (x - mu).pow(2).mean(1, keepdim=True)  # 计算方差
        y = (x - mu) / (var + eps).sqrt()  # 标准化
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)  # 应用权重和偏置
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)

        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))  # 权重参数
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))  # 偏置参数
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)  # 调用自定义的LayerNormFunction

class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)  # 第一个卷积层
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)  # 第二个卷积层

        initialize_weights([self.conv1, self.conv2], 0.1)  # 初始化权重

    def forward(self, x):
        identity = x  # 保存输入以便残差连接
        out = F.relu(self.conv1(x), inplace=True)  # 第一个卷积后激活
        out = self.conv2(out)  # 第二个卷积
        return identity + out  # 返回加上输入的结果

class BIE(nn.Module):
    def __init__(self, nf=64):
        super(BIE, self).__init__()

        self.conv1 = ResidualBlock_noBN(nf)  # 第一个残差块
        self.conv2 = self.conv1  # 第二个残差块，结构相同
        self.convf1 = nn.Conv2d(nf * 2, nf, 1, 1, padding=0)  # 1x1卷积
        self.convf2 = self.convf1  # 另一个1x1卷积

        self.scale = nf ** -0.5  # 缩放因子
        self.norm_s = LayerNorm2d(nf)  # 层归一化
        self.clustering = nn.Conv2d(nf, nf, 1, 1, padding=0)  # 聚类卷积
        self.unclustering = nn.Conv2d(nf * 2, nf, 1, stride=1, padding=0)  # 解聚类卷积

        self.v1 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)  # 特征映射卷积
        self.v2 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)  # 特征映射卷积

        initialize_weights([self.convf1, self.convf2, self.clustering, self.unclustering, self.v1, self.v2], 0.1)  # 初始化权重

    def forward(self, x_1, x_2, x_s):
        b, c, h, w = x_1.shape

        # 两侧Conv2D
        x_1_ = self.conv1(x_1)  # 通过第一个残差块
        x_2_ = self.conv2(x_2)  # 通过第二个残差块


        # 计算共享类别中心
        shared_class_center1 = self.clustering(self.norm_s(self.convf1(torch.cat([x_s, x_2], dim=1)))).view(b, c, -1)
        shared_class_center2 = self.clustering(self.norm_s(self.convf2(torch.cat([x_s, x_1], dim=1)))).view(b, c, -1)

        # 中左 中右
        v_1 = self.v1(x_1).view(b, c, -1).permute(0, 2, 1)  # 特征映射和转置
        v_2 = self.v2(x_2).view(b, c, -1).permute(0, 2, 1)  # 特征映射和转置
        # 计算注意力
        att1 = torch.bmm(shared_class_center1, v_1) * self.scale
        att2 = torch.bmm(shared_class_center2, v_2) * self.scale
        # 计算输出
        out_1 = torch.bmm(torch.softmax(att1, dim=-1), v_1.permute(0, 2, 1)).view(b, c, h, w)
        out_2 = torch.bmm(torch.softmax(att2, dim=-1), v_2.permute(0, 2, 1)).view(b, c, h, w)

        x_s_ = self.unclustering(torch.cat([shared_class_center1.view(b, c, h, w), shared_class_center2.view(b, c, h, w)], dim=1)) + x_s

        return out_1 + x_2_, out_2 + x_1_, x_s_  # 返回结果

if __name__ == '__main__':
    x1 = torch.randn(1, 32, 64, 64)  # 生成随机输入张量1
    x2 = torch.randn(1, 32, 64, 64)  # 生成随机输入张量2
    x3 = torch.randn(1, 32, 64, 64)  # 生成随机输入张量3

    Model = BIE(nf=32)  # 初始化BIE模型
    out = Model(x1, x2, x3)  # 计算输出

    print("Input size:", x1.size(), x2.shape, x3.shape)  # 打印输入尺寸
    print("Output size:", out[0].size())  # 打印输出尺寸1
    print("Output size:", out[1].size())  # 打印输出尺寸2
    print("Output size:", out[2].size())  # 打印输出尺寸3
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
