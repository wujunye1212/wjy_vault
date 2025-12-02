import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2403.01105
    论文题目：Depth Information Assisted Collaborative Mutual Promotion Network for Single Image Dehazing（CVPR 2024）
    中文题目：用于单图像去雾的深度信息辅助协同互促网络（CVPR 2024）
    讲解视频：https://www.bilibili.com/video/BV1npdzYoEUi/
    调制融合模块模块（Modulation Fusion Module, MFM）：
        实际意义：①多特征利用问题：去雾过程需要利用多种特征信息，而不同特征在去雾中的重要性不同。
                ②特征表示能力问题：单图像去雾需要网络准确捕捉雾图的各种特征。
        实现方式：①通过动态调整融合权重，实现针对不同特征有效融合。
                ②促进通道间特征交互，增强雾图特征的表示能力，理解物体边缘和纹理特征。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】   
"""
class Modulation_Fusion_Module(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(Modulation_Fusion_Module, self).__init__()

        # 保存 height 参数，height 表示特征图的分组数
        self.height = height
        # 计算中间层的维度 d，取 dim 除以 reduction 的结果和 4 中的最大值
        d = max(int(dim / reduction), 4)

        # 定义自适应平均池化层，将输入特征图池化到 1x1 大小
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义一个顺序容器，包含一系列的卷积层和激活函数
        self.mlp = nn.Sequential(
            # 第一个卷积层，输入通道数为 dim，输出通道数为 d，卷积核大小为 1
            nn.Conv2d(dim, d, 1, bias=False),
            # ReLU 激活函数，增加模型的非线性
            nn.ReLU(),
            # 第二个卷积层，输入通道数为 d，输出通道数为 dim * height，卷积核大小为 1
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        # 定义 Softmax 激活函数，用于在维度 1 上进行归一化
        self.softmax = nn.Softmax(dim=1)

    # 前向传播方法，定义了模块的前向计算逻辑
    def forward(self, in_feats1, in_feats2):
        # 将输入的两个特征图存储在一个列表中
        in_feats = [in_feats1, in_feats2]
        # 获取输入特征图的批次大小 B、通道数 C、高度 H 和宽度 W
        B, C, H, W = in_feats[0].shape

        # 沿着通道维度将两个输入特征图拼接在一起
        in_feats = torch.cat(in_feats, dim=1)
        # 调整输入特征图的形状，将其按照 height 进行分组
        in_feats = in_feats.view(B, self.height, C, H, W)

        # 沿着 height 维度对输入特征图进行求和
        feats_sum = torch.sum(in_feats, dim=1)
        # 对求和后的特征图进行自适应平均池化，然后通过 MLP 网络
        attn = self.mlp(self.avg_pool(feats_sum))
        # 调整注意力图的形状，并通过 Softmax 函数进行归一化
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        # 将输入特征图与注意力图逐元素相乘，然后沿着 height 维度求和
        out = torch.sum(in_feats * attn, dim=1)
        # 返回最终的输出特征图
        return out

if __name__ == "__main__":
    input1 = torch.randn(1, 64, 50, 50)
    input2 = torch.randn(1, 64, 50, 50)
    model = Modulation_Fusion_Module(64)
    output = model(input1, input2)
    print(f"Input1 Shape: {input1.shape}")
    print(f"Input2 Shape: {input2.shape}")
    print(f"Output Shape: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")