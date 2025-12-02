import torch
import torch.nn as nn
import torch.nn.functional as F

'''
论文地址：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf
论文题目：SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution（ECCV 2024）
讲解视频：https://www.bilibili.com/video/BV1tiwBeSErk/
    基于部分卷积的前馈网络（Partial convolution-based feed-forward network,PCFN）
    
       常规前馈网络（FFN）[12, 40]对每个像素位置的操作是相同的，缺乏空间维度的信息交换，提出了一种高效的基于卷积的前馈网络（PCFN）
       
       首先，使用带有GELU 激活函数的 1×1 卷积对扩展的隐藏空间进行跨信道交互。然后，它将隐藏特征分割成 {F1ρ , F2ρ } 两块，并采用3×3卷积
            和 GELU 激活来处理 F1ρ，以编码局部上下文信息。然后将处理后的 F1ρ 和 F2ρ 合并，并输入 1×1 卷积，以进一步混合特征，并且还原为原始维度。
'''


# 定义一个名为PCFN的神经网络模块，继承自nn.Module
class PCFN(nn.Module):
    # 初始化方法，设置模型的基本参数
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()  # 调用父类nn.Module的初始化方法

        # 计算隐藏层维度，基于输入维度dim和增长因子growth_rate
        hidden_dim = int(dim * growth_rate)

        # 计算部分处理维度p_dim，基于隐藏层维度hidden_dim和比例p_rate
        p_dim = int(hidden_dim * p_rate)

        # 第一个卷积层，将输入从dim维度变换到hidden_dim维度
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        # 第二个卷积层，对p_dim维度进行3x3卷积操作
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)

        # 使用GELU作为激活函数
        self.act = nn.GELU()
        # 第三个卷积层，将hidden_dim维度变换回dim维度
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        # 将计算得到的p_dim存储为成员变量
        self.p_dim = p_dim
        # 将计算得到的hidden_dim存储为成员变量
        self.hidden_dim = hidden_dim

    # 前向传播方法，定义数据如何通过网络
    def forward(self, x):
        # 对输入x应用第一个卷积层，并使用GELU激活
        x = self.act(self.conv_0(x))

        # 根据之前计算的p_dim将x分割成两部分：x1和x2
        x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)

        # 对x1应用第二个卷积层并激活
        x1 = self.act(self.conv_1(x1))

        # 将处理后的x1与未经处理的x2沿通道维度拼接，然后通过第三个卷积层
        x = self.conv_2(torch.cat([x1, x2], dim=1))

        # 返回最终输出
        return x

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1,32,256,256)
    model = PCFN(dim=32)
    output = model (input)
    print('input_size:', input.size())
    print('output_size:', output.size())