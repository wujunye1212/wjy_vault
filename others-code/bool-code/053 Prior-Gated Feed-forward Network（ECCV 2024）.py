import torch
import torch.nn as nn
'''
    论文地址：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05751.pdf
    论文题目：Efficient Frequency-Domain Image Deraining with Contrastive Regularization （ECCV 2024）
    中文题目： 利用对比正则化实现高效的频域图像去雨（ECCV 2024）
    讲解视频：https://www.bilibili.com/video/BV1MLUnYPEdQ/
        先验门控前馈网络（Prior-Gated Feed-forward Network,PGFN) ：
           依据：以门控方式将先验知识集成到 FFN 中，以改善局部细节恢复。
           优点：1、残差通道先验，它有效地保留清晰的结构。
                2、上半部分：拓宽通道维度，然后使用 DConv 细化局部细节。
                3、下半部分：RCP 特征作为分组门控权重，用来强化背景信息。
'''
class Prior_Gated_Feed_forward_Network(nn.Module):
    def __init__(
            self,
            dim,
            se_ratio=4,
            local_size=8,
            scale_ratio=2,
            spilt_num=4
    ):
        super(Prior_Gated_Feed_forward_Network, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim * scale_ratio // spilt_num

        # 初始化卷积层，使用点卷积和GELU激活
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )

        # 最终卷积层，使用点卷积和GELU激活
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU()
        )

        # 深度可分离卷积层，卷积核大小为3
        self.conv_dw = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=3 // 2, groups=dim * 2,
                      padding_mode='reflect'),
            nn.GELU()
        )

        # Mask输入处理层，使用点卷积和GELU激活
        self.mask_in = nn.Sequential(
            nn.Conv2d(1, self.dim_sp, 1),
            nn.GELU()
        )

        # 深度可分离卷积层1，卷积核大小为3，输出使用Sigmoid激活
        self.mask_dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp // 2, 1, kernel_size=3, padding=3 // 2, padding_mode='reflect'),
            nn.Sigmoid()
        )

        # 深度可分离卷积层2，卷积核大小为5，输出使用Sigmoid激活
        self.mask_dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp // 2, 1, kernel_size=5, padding=5 // 2, padding_mode='reflect'),
            nn.Sigmoid()
        )

        # Mask输出处理层，使用点卷积和GELU激活
        self.mask_out = nn.Sequential(
            nn.Conv2d(2, 1, 1),
            nn.GELU()
        )

    def forward(self, x, mask):
        # 通过初始卷积层
        x = self.conv_init(x)
        # 通过深度可分离卷积层
        x = self.conv_dw(x)
        # 将特征图分割为两部分
        x = list(torch.split(x, self.dim, dim=1))

        # 处理输入的mask
        mask = self.mask_in(mask)
        # 将mask分割为两部分
        mask = list(torch.split(mask, self.dim_sp // 2, dim=1))
        # 通过深度可分离卷积层1
        mask[0] = self.mask_dw_conv_1(mask[0])
        # 通过深度可分离卷积层2
        mask[1] = self.mask_dw_conv_2(mask[1])

        # 应用mask到特征图
        x[0] = mask[0] * x[0]
        x[1] = mask[1] * x[1]

        # 合并特征图
        x = torch.cat(x, dim=1)
        # 通过最终卷积层
        x = self.conv_fina(x)
        # 合并mask并通过输出处理层
        mask = self.mask_out(torch.cat(mask, dim=1))

        return x, mask


if __name__ == "__main__":
    # 创建随机输入张量和mask张量
    input_tensor = torch.randn(1, 64, 32, 32)
    mask_tensor = torch.randn(1, 1, 32, 32)

    # 初始化网络
    network = Prior_Gated_Feed_forward_Network(dim=64)

    # 前向传播
    output, mask_output = network(input_tensor, mask_tensor)

    # 打印输出形状
    print("Output shape:", output.shape)
    print("Mask output shape:", mask_output.shape)


    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息
