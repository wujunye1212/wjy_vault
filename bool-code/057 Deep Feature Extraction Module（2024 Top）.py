import torch
import torch.nn as nn

'''
    论文地址：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05751.pdf
    论文题目：B2CNet: A Progressive Change Boundary-to-Center Refinement Network for Multitemporal Remote Sensing Images Change Detection （2024 Top）
    中文题目：用于多时相遥感图像变化检测的渐进式变化边界到中心细化网络（2024 Top）
    讲解视频：https://www.bilibili.com/video/BV1vxUdYVE4C/
        深度特征提取模块（Deep Feature Extraction Module, DFEM），：
           优点：高级语义特征在准确定位和识别变化区域方面起着至关重要的作用。另一方面，详细的纹理特征提供了更精确的边界和纹理信息。
           步骤：首先分别对CBM和BFAM的输入特征进行连接和求和。为了减少计算量，采用1×1卷积将通道数减少一半。
                    此外，为了保持信息完整性，使用残差连接将组合信息与其相乘。接下来，使用3×3卷积块提取深度特征。
                    为了最大限度地减少信息丢失，在RELU之前选择残差连接。最后，将上一级 CBM 的输出特征与深度特征相加
'''
# DFEM模块，用于特征融合和增强
class DFEM(nn.Module):
    def __init__(self, inc, outc):
        super(DFEM, self).__init__()

        # 1x1卷积层，用于特征压缩
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

        # 3x3卷积层，用于特征提取
        self.Conv = nn.Sequential(
            nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)
        # 上采样层，用于增加特征图的分辨率
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, diff, accom):
        # 将两个输入特征图拼接
        cat = torch.cat([accom, diff], dim=1)
        # 通过1x1卷积层
        cat = self.Conv_1(cat) + diff + accom
        # 通过3x3卷积层
        c = self.Conv(cat) + cat
        # 激活并加上差异特征
        c = self.relu(c) + diff
        # 上采样
        c = self.Up(c)
        return c

if __name__ == "__main__":
    # 定义输入通道数和输出通道数
    inc = 8
    outc = 8
    # 初始化模型
    model = DFEM(2 * inc, outc)

    # 创建随机输入张量
    diff = torch.randn(1, inc, 32, 32)  # 批量大小为1，输入通道数为inc，尺寸为32x32
    accom = torch.randn(1, inc, 32, 32)  # 与diff具有相同的尺寸

    # 通过模型进行前向传播
    output = model(diff, accom)

    # 打印输出张量的形状
    print("Output shape:", output.shape)
    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息
