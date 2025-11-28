from einops import rearrange
import torch
from torch.nn import functional as F
from torch import nn
'''
    论文地址：https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_DNF_Decouple_and_Feedback_Network_for_Seeing_in_the_Dark_CVPR_2023_paper.pdf
    论文题目：DNF: Decouple and Feedback Network for Seeing in the Dark （CVPR 2023）
    中文题目：DNF：在黑暗中看到的解耦和反馈
    讲解视频：https://www.bilibili.com/video/BV1cFUPYgE3P/
        通道独立去噪（Channel-Independent ，CID）：
           依据：1、RAW 格式的低光图像存在信号独立噪声。
                2、RAW 域中不同通道信号本身相关性低，导致噪声分布在通道间倾向于独立。在去噪过程中要防止通道间信息交换，以处理这种通道独立的噪声分布。
                3、设置残差开关，为了适应去噪阶段和颜色恢复阶段不同的需求，例如在去噪阶段更好地估计噪声，在颜色恢复阶段准确地重建信号等。
           特点：大核深度卷积（7×7）用于去噪操作
'''

class DConv7(nn.Module):
    def __init__(self, f_number, padding_mode='reflect') -> None:
        # 初始化 DConv7 类，f_number 表示特征图通道数，padding_mode 表示填充模式
        super().__init__()
        # 定义一个深度可分离卷积层，使用 7x7 的卷积核，填充为 3，组数为 f_number，实现每个通道独立卷积
        self.dconv = nn.Conv2d(f_number, f_number, kernel_size=7, padding=3, groups=f_number, padding_mode=padding_mode)

    def forward(self, x):
        # 前向传播函数，应用深度卷积
        return self.dconv(x)

class MLP(nn.Module):
    def __init__(self, f_number, excitation_factor=2) -> None:
        # 初始化 MLP 类，f_number 表示特征图通道数，excitation_factor 表示扩展因子
        super().__init__()
        self.act = nn.GELU()  # 使用 GELU 激活函数
        # 定义第一个逐点卷积层，将通道数扩展为 excitation_factor 倍
        self.pwconv1 = nn.Conv2d(f_number, excitation_factor * f_number, kernel_size=1)
        # 定义第二个逐点卷积层，将通道数还原为原始通道数
        self.pwconv2 = nn.Conv2d(f_number * excitation_factor, f_number, kernel_size=1)

    def forward(self, x):
        # 前向传播函数
        x = self.pwconv1(x)  # 应用第一个逐点卷积
        x = self.act(x)      # 应用激活函数
        x = self.pwconv2(x)  # 应用第二个逐点卷积
        return x

class CID(nn.Module):
    def __init__(self, f_number):
        # 初始化 CID 类，f_number 表示特征图通道数
        super().__init__()
        self.channel_independent = DConv7(f_number)  # 定义通道独立的卷积层
        self.channel_dependent = MLP(f_number)       # 定义通道依赖的 MLP 层

    def forward(self, x):
        # 前向传播函数，先应用通道独立卷积，再应用通道依赖 MLP
        return self.channel_dependent(self.channel_independent(x))

if __name__ == '__main__':
    # 主函数，测试 CID 模型
    input = torch.randn(1, 64, 128, 128)  # 生成随机输入张量，形状为 (1, 64, 128, 128)
    model = CID(64)  # 创建 CID 模型实例，特征图通道数为 64
    output = model(input)  # 通过模型计算输出
    print(f"input shape: {input.shape}")  # 打印输入张量的形状
    print(f"output shape: {output.shape}")  # 打印输出张量的形状


    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息
