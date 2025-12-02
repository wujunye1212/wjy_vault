import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/abs/2403.10778
    论文题目：HCF-Net: Hierarchical Context Fusion Network for Infrared Small Object Detection（arxiv 2024）
    中文题目：HCF-Net：用于红外小物体检测的分层上下文融合网络（arxiv 2024）
    讲解视频：https://www.bilibili.com/video/BV1hvqbYAEiQ/
        多扩张通道细化模块（Multi-Dilated Channel Refiner ,MDCR）：
            作用：通过多个深度可分离卷积层以及不同的卷积率捕获不同感受野范围的空间特征。
"""
class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups = 1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups = groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class MDCR(nn.Module):
    def __init__(self, in_features, out_features, norm_type='bn', activation=True, rate=[1, 6, 12, 18]):
        super().__init__()

        self.block1 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[0],
            dilation=rate[0],
            norm_type=norm_type,
            activation=activation,
            groups=in_features // 4
            )
        self.block2 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            activation=activation,
            groups=in_features // 4
            )
        self.block3 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[2],
            dilation=rate[2],
            norm_type=norm_type,
            activation=activation,
            groups=in_features // 4
            )
        self.block4 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[3],
            dilation=rate[3],
            norm_type=norm_type,
            activation=activation,
            groups=in_features // 4
            )
        self.out_s = conv_block(
            in_features=4,
            out_features=4,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
        )
        self.out = conv_block(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
            )

    def forward(self, x):
        split_tensors = []
        x = torch.chunk(x, 4, dim=1)
        x1 = self.block1(x[0])
        x2 = self.block2(x[1])
        x3 = self.block3(x[2])
        x4 = self.block4(x[3])
        for channel in range(x1.size(1)):
            channel_tensors = [tensor[:, channel:channel + 1, :, :] for tensor in [x1, x2, x3, x4]]
            concatenated_channel = self.out_s(torch.cat(channel_tensors, dim=1))  # 拼接在 batch_size 维度上
            split_tensors.append(concatenated_channel)
        x = torch.cat(split_tensors, dim=1)
        x = self.out(x)
        return x


if __name__ == '__main__':
    # 创建一个随机输入张量作为示例
    input_tensor = torch.randn(1, 64, 32, 32)  # 假设输入尺寸为 [batch_size, channels, height, width]

    # 实例化 MDCR 模块
    mdcr = MDCR(in_features=64, out_features=64)

    # 将输入张量传递给 MDCR 模块并获取输出
    output_tensor = mdcr(input_tensor)

    # 打印输入和输出的形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
