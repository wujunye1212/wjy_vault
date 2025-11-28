import torch
import torch.nn as nn
'''
    论文地址：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05751.pdf
    论文题目：B2CNet: A Progressive Change Boundary-to-Center Refinement Network for Multitemporal Remote Sensing Images Change Detection （2024 Top）
    中文题目：用于多时相遥感图像变化检测的渐进式变化边界到中心细化网络（2024 Top）
    讲解视频：https://www.bilibili.com/video/BV1cYSKYJEDf/
           边界变化感知模块（Change Boundary-Aware Module, CBM) ：
           优点：特征差异操作能够有效地描绘变化区域的边界和轮廓。
           步骤：首先，使用SimAM注意力机制对输入的特征进行预处理，以识别值得关注的区域。
                然后，通过池化、减法和卷积等操作提取边缘特征，增强边缘信息的对比度和显著性。
                接着，对边缘增强的特征进行特征差异计算，以获得更丰富的上下文信息，从而更准确地定位变化区域。
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

        # 计算每个像素与均值的差的平方
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # 计算激活函数输入
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        # 返回加权后的输入
        return x * self.activaton(y)


# 差异模块，用于特征差异提取
class diff_moudel(nn.Module):
    def __init__(self, in_channel):
        super(diff_moudel, self).__init__()
        # 平均池化层
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        # 1x1卷积层
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(in_channel)
        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
        # SIMAM模块
        self.simam = simam_module()

    def forward(self, x):
        # 应用SIMAM模块
        x = self.simam(x)
        # 计算边缘特征
        edge = x - self.avg_pool(x)
        # 计算权重
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        # 计算输出
        out = weight * x + x
        # 再次应用SIMAM模块
        out = self.simam(out)
        return out


# CBM模块，用于特征融合
class CBM(nn.Module):
    def __init__(self, in_channel):
        super(CBM, self).__init__()
        # 差异模块1
        self.diff_1 = diff_moudel(in_channel)
        # 差异模块2
        self.diff_2 = diff_moudel(in_channel)
        # SIMAM模块
        self.simam = simam_module()

    def forward(self, x1, x2):
        # 计算两个输入的差异特征
        d1 = self.diff_1(x1)
        d2 = self.diff_2(x2)
        # 计算绝对差异
        d = torch.abs(d1 - d2)
        # 应用SIMAM模块
        d = self.simam(d)
        return d


if __name__ == "__main__":
    # 创建随机输入张量
    input1 = torch.randn(1, 32, 128, 128)
    input2 = torch.randn(1, 32, 128, 128)

    # 初始化CBM模块
    cbm = CBM(32)

    output = cbm(input1, input2)

    # 打印输入和输出的形状
    print('CBM_input_size:', input1.size())
    print('CBM_output_size:', output.size())
