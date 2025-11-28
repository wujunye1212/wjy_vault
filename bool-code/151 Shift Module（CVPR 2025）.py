import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2503.02394
    论文题目：BHViT: Binarized Hybrid Vision Transformer（CVPR 2025）
    中文题目：BHViT：二值化混合视觉Transformer（CVPR 2025）
    讲解视频：https://www.bilibili.com/video/BV1qQ53z7Evo/
        移位模块（Shift Module）：
            实际意义：①特征损失问题：在二进制模型中，数据在经过一系列二值化操作和线性层计算后，信息容易丢失。注：二值化是指原本丰富连续数值信息被简化为二值。
            实现方式：代码为准，很简单
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class Shift_Module(nn.Module):
    # 初始化方法，接收shift_size（滚动步长）作为参数，默认值为1
    def __init__(self, shift_size=1):
        # 调用父类nn.Module的初始化方法（必须操作）
        super(Shift_Module, self).__init__()
        # 将传入的滚动步长保存为实例变量
        self.shift_size = shift_size

    def forward(self, x):
        # 将输入张量x沿通道维度（dim=1）均分为4个部分
        # 假设输入通道数为4的倍数（例如32通道会被分为4个8通道的子张量）
        x1, x2, x3, x4 = x.chunk(4, dim=1)

        # 对x1张量沿高度维度（dim=2）正向滚动shift_size步
        # 例如：如果高度是50，向上滚动1步，第一行变为最后一行
        x1 = torch.roll(x1, self.shift_size, dims=2)

        # 对x2张量沿高度维度（dim=2）反向滚动shift_size步
        # 例如：向下滚动1步，最后一行变为第一行
        x2 = torch.roll(x2, -self.shift_size, dims=2)

        # 对x3张量沿宽度维度（dim=3）正向滚动shift_size步
        # 例如：如果宽度是50，向左滚动1步，第一列变为最后一列
        x3 = torch.roll(x3, self.shift_size, dims=3)

        # 对x4张量沿宽度维度（dim=3）反向滚动shift_size步
        # 例如：向右滚动1步，最后一列变为第一列
        x4 = torch.roll(x4, -self.shift_size, dims=3)

        # 将处理后的4个子张量沿通道维度（dim=1）重新拼接成完整张量
        x = torch.cat([x1, x2, x3, x4], 1)
        return x

if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 50, 50)
    model = Shift_Module()
    output = model(input_tensor)
    print(f'Input size: {input_tensor.size()}')
    print(f'Output size: {output.size()}')
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")