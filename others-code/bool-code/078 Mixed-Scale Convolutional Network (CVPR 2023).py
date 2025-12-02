import torch
import torch.nn as nn

"""
    论文地址：https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Learning_a_Sparse_Transformer_Network_for_Effective_Image_Deraining_CVPR_2023_paper.pdf
    论文题目：Learning A Sparse Transformer Network for Effective Image Deraining（CVPR 2023）
    中文题目：学习稀疏的变压器网络以实现有效的图像去雨 （CVPR 2023）
    讲解视频：https://www.bilibili.com/video/BV1jpkjYHExw
        混合尺度卷积网络（Mixed-Scale Convolutional Network，MSFN）：
            设计目的：先前研究在常规前馈网络中引入单尺度深度卷积来改进局部性，但忽略多尺度雨纹的相关性。
            理论支撑：通过添加不同尺度的局部特征提取和融合。
"""

class Mixed_Scale_Feedforward_Network(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(Mixed_Scale_Feedforward_Network, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)  # 计算隐藏层特征数

        # 定义输入投影层
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 定义3x3和5x5的深度可分离卷积
        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()  # 3x3卷积后的ReLU激活
        self.relu5 = nn.ReLU()  # 5x5卷积后的ReLU激活

        # 定义第二层3x3和5x5的深度可分离卷积
        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features, bias=bias)

        self.relu3_1 = nn.ReLU()  # 第二层3x3卷积后的ReLU激活
        self.relu5_1 = nn.ReLU()  # 第二层5x5卷积后的ReLU激活

        # 定义输出投影层
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)  # 输入投影

        # 通过3x3卷积和ReLU激活，然后分块
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        # 通过5x5卷积和ReLU激活，然后分块
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        # 将3x3和5x5的输出拼接
        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        # 通过第二层3x3卷积和ReLU激活
        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        # 通过第二层5x5卷积和ReLU激活
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        # 将两个输出拼接
        x = torch.cat([x1, x2], dim=1)

        # 输出投影
        x = self.project_out(x)

        return x

if __name__ == '__main__':
    # 创建一个随机输入张量作为示例
    input_tensor = torch.randn(1, 64, 32, 32)  # 假设输入尺寸为 [batch_size, channels, height, width]

    # 实例化 Mixed_Scale_Feedforward_Network 模块
    mdcr = Mixed_Scale_Feedforward_Network(dim=64, ffn_expansion_factor=2, bias=True)

    # 将输入张量传递给 Mixed_Scale_Feedforward_Network 模块并获取输出
    output_tensor = mdcr(input_tensor)

    # 打印输入和输出的形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")