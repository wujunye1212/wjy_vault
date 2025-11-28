from einops import rearrange
import torch
from torch.nn import functional as F
from torch import nn

'''
    论文地址：https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_DNF_Decouple_and_Feedback_Network_for_Seeing_in_the_Dark_CVPR_2023_paper.pdf
    论文题目：DNF: Decouple and Feedback Network for Seeing in the Dark （CVPR 2023）
    中文题目：DNF：在黑暗中看到的解耦和反馈
    讲解视频：https://www.bilibili.com/video/BV18pmXYREND/
        门控融合模块（Gated Fusion Modules，GFM）：
                作用：自适应地将噪声估计与初始去噪特征进行融合。在特征门控过程中，将有益信息能在空间和通道维度上被自适应地选择和合并。
'''
class GFM(nn.Module):
    def __init__(self, in_channels, feature_num=2, bias=True, padding_mode='reflect', **kwargs) -> None:
        # 初始化 GFM 类，in_channels 表示输入通道数，feature_num 表示特征数量
        super().__init__()
        self.feature_num = feature_num  # 保存特征数量

        hidden_features = in_channels * feature_num  # 计算隐藏特征数
        self.pwconv = nn.Conv2d(hidden_features, hidden_features * 2, 1, 1, 0, bias=bias)  # 逐点卷积，扩展通道数
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=hidden_features * 2)  # 深度卷积
        self.project_out = nn.Conv2d(hidden_features, in_channels, kernel_size=1, bias=bias)  # 输出投影层
        self.mlp = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)  # MLP 层

    def forward(self, *inp_feats):
        """
            改进：https://space.bilibili.com/346680886/search/video?keyword=%E9%97%A8%E6%8E%A7
        """
        assert len(inp_feats) == self.feature_num  # 确保输入特征数量与定义的一致
        shortcut = inp_feats[0]             # 保存第一个输入

        x = torch.cat(inp_feats, dim=1)     # 在通道维度上拼接输入特征
        x = self.pwconv(x)                      # 应用逐点卷积
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 应用深度卷积并分块
        x = F.gelu(x1) * x2                     # GELU 激活后与另一块相乘
        x = self.project_out(x)                 # 应用输出投影

        return self.mlp(x + shortcut)  # 加上捷径连接并通过 MLP 层

if __name__ == '__main__':
    in_channels = 64  # 输入通道数
    input_shape = (1, in_channels, 32, 32)  # 假设输入张量的形状

    input1 = torch.randn(input_shape)  # 生成随机输入张量 input1
    input2 = torch.randn(input_shape)  # 生成随机输入张量 input2

    # 合并的特征数 【理论上无限合并】 多尺度融合
    feature_num = 2
    model = GFM(in_channels=in_channels, feature_num=feature_num)  # 创建 GFM 模型实例

    # 前向传播
    output = model(input1, input2)  # 通过模型计算输出

    print(f"input1 shape: {input1.shape}")  # 打印输入张量1的形状
    print(f"input2 shape: {input2.shape}")  # 打印输入张量2的形状
    print(f"output shape: {output.shape}")  # 打印输出张量的形状

    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息

