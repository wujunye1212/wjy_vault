import torch
import torch.nn as nn
'''
    论文地址：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05751.pdf
    论文题目：B2CNet: A Progressive Change Boundary-to-Center Refinement Network for Multitemporal Remote Sensing Images Change Detection （2024 Top）
    中文题目：用于多时相遥感图像变化检测的渐进式变化边界到中心细化网络（2024 Top）
    讲解视频：https://www.bilibili.com/video/BV121U2YBERx/
        双时态特征聚合模块（Bitemporal Feature Aggregation Module, BFAM) ：
           优点：双时态特征由于其固有的性质，在展现丰富细节的同时还能保留空间关系。
           步骤：①对双时态特征进行通道拼接和不同扩张率的并行扩张组卷积来提取特征，实现通过不同的感受野捕捉各种大小的变化区域，
                        同时利用分组卷积保持特征的空间完整性。
                ②将四个卷积输出进行通道连接，并使用1×1卷积块进行通道下采样，进一步细化特征，此过程中还利用了SimAM注意力机制。
                        然后，分别计算不同时态特征的像素级特征重要性，并将提取的共同特征与各自的时态特征相乘得到相似度。
                ③将低层细节特征与高级变化特征相加，并通过SimAM注意力机制获得最终的聚合特征
'''
class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()
        # 使用Sigmoid激活函数
        self.activaton = nn.Sigmoid()
        # 设置lambda参数
        self.e_lambda = e_lambda

    def __repr__(self):
        # 返回模块的名称和lambda值
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        # 返回模块名称
        return "simam"

    def forward(self, x):
        # 获取输入张量的尺寸
        b, c, h, w = x.size()
        n = w * h - 1
        # 计算每个元素与均值的差的平方
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # 计算y值
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # 返回经过激活和乘法后的结果
        return x * self.activaton(y)

# 定义BFAM模块类
class BFAM(nn.Module):
    def __init__(self, inp, out):
        super(BFAM, self).__init__()

        # 初始化SimAM模块
        self.pre_siam = simam_module()
        self.lat_siam = simam_module()

        # 设置输出通道数
        out_1 = inp
        inp = inp + out
        # 定义多个卷积层，使用不同的膨胀率
        self.conv_1 = nn.Conv2d(inp, out_1, padding=1, kernel_size=3, groups=out_1, dilation=1)
        self.conv_2 = nn.Conv2d(inp, out_1, padding=2, kernel_size=3, groups=out_1, dilation=2)
        self.conv_3 = nn.Conv2d(inp, out_1, padding=3, kernel_size=3, groups=out_1, dilation=3)
        self.conv_4 = nn.Conv2d(inp, out_1, padding=4, kernel_size=3, groups=out_1, dilation=4)

        # 定义融合层
        self.fuse = nn.Sequential(
            nn.Conv2d(out_1 * 4, out_1, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_1),
            nn.ReLU(inplace=True)
        )

        # 初始化SimAM模块
        self.fuse_siam = simam_module()

        # 定义输出层
        self.out = nn.Sequential(
            nn.Conv2d(out_1, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )

    def forward(self, inp1, inp2):
        # 初始化最后特征为None
        last_feature = None

        # 合并输入
        x = torch.cat([inp1, inp2], dim=1)

        # 通过多个卷积层处理输入
        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)
        c4 = self.conv_4(x)
        # 合并卷积结果
        cat = torch.cat([c1, c2, c3, c4], dim=1)
        # 通过融合层
        fuse = self.fuse(cat)

        # 处理输入特征
        inp1_siam = self.pre_siam(inp1)
        inp2_siam = self.lat_siam(inp2)
        inp1_mul = torch.mul(inp1_siam, fuse)
        inp2_mul = torch.mul(inp2_siam, fuse)

        # 通过SimAM模块处理融合特征
        fuse = self.fuse_siam(fuse)

        # 根据last_feature的状态进行输出计算
        if last_feature is None:
            out = self.out(fuse + inp1 + inp2 + inp2_mul + inp1_mul)
        else:
            out = self.out(fuse + inp2_mul + inp1_mul + last_feature + inp1 + inp2)

        # 通过SimAM模块处理输出
        out = self.fuse_siam(out)

        return out

if __name__ == "__main__":
    # 创建输入张量
    input1 = torch.randn(1, 30, 128, 128)
    input2 = torch.randn(1, 30, 128, 128)
    # 初始化BFAM模块
    bfam = BFAM(30, 30)
    # 计算输出
    output = bfam(input1, input2)
    # 打印输入和输出尺寸
    print('BFAM_input_size:', input1.size())
    print('BFAM_output_size:', output.size())
    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息
