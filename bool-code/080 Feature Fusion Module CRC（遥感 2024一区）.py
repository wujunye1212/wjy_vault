import torch
import torch.nn as nn

"""
    论文地址：https://ieeexplore.ieee.org/abstract/document/10423050
    论文题目：FFCA-YOLO for Small Object Detection in Remote Sensing Images（2024 一区TOP）
    中文题目：遥感图像小目标检测的FFCA-YOLO （2024 一区TOP）
    讲解视频：https://www.bilibili.com/video/BV1pikMYzE7t/
        特征融合模块（Feature Fusion Module，FFM）：
            理论支撑：通过使用通道重加权策略，根据不同特征图之间的差异程度，动态调整通道重要性，从而更好地融合不同尺度的特征信息，以更好地表示小目标。
"""

class FFM_Concat2(nn.Module):
    # 特征融合模块，用于连接两个特征图 (Feature Fusion Module)
    def __init__(self, dimension=1, Channel1=1, Channel2=1):
        super(FFM_Concat2, self).__init__()
        self.d = dimension  # 拼接维度
        self.Channel1 = Channel1  # 第一个输入的通道数
        self.Channel2 = Channel2  # 第二个输入的通道数
        self.Channel_all = int(Channel1 + Channel2)  # 总通道数
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)  # 可学习的权重参数，初始化为1
        self.epsilon = 0.0001  # 防止除零的小值

    def forward(self, x):
        N1, C1, H1, W1 = x[0].size()  # 获取第一个输入的维度
        N2, C2, H2, W2 = x[1].size()  # 获取第二个输入的维度

        w = self.w[:(C1 + C2)] #  确保权重数量与输入通道数匹配, 尤其在剪枝后
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 权重归一化

        x1 = (weight[:C1] * x[0].view(N1, H1, W1, C1)).view(N1, C1, H1, W1) # 对第一个输入进行加权
        x2 = (weight[C1:] * x[1].view(N2, H2, W2, C2)).view(N2, C2, H2, W2) # 对第二个输入进行加权
        x = [x1, x2] # 将加权后的特征图重新组成列表
        return torch.cat(x, self.d)  # 在指定维度上拼接

if __name__ == '__main__':

    batch_size = 4  # 批量大小
    in_channels_1 = 64  # 第一个输入的通道数
    in_channels_2 = 32  # 第二个输入的通道数
    height = 32  # 高度
    width = 32  # 宽度

    # 创建两个输入张量，注意通道数可以不同
    input1 = torch.randn(batch_size, in_channels_1, height, width)  # 创建第一个随机输入张量
    input2 = torch.randn(batch_size, in_channels_2, height, width)  # 创建第二个随机输入张量

    inputs = [input1, input2]  # 将两个输入张量放入列表
    ffm = FFM_Concat2(dimension=1, Channel1=in_channels_1, Channel2=in_channels_2)  # 创建FFM_Concat2模块实例
    output = ffm(inputs)  # 前向传播

    print("Input1 shape:", input1.shape)  # 打印第一个输入的形状
    print("Input2 shape:", input2.shape)  # 打印第二个输入的形状
    print("Output shape:", output.shape)  # 打印输出的形状
