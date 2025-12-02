import torch
from torch import nn
from torch.nn import functional as F

'''
    论文地址：https://dl.acm.org/doi/abs/10.1145/3664647.3680650
    论文题目：Multi-Scale and Detail-Enhanced Segment Anything Model for Salient Object Detection（A会议 2024）
    中文题目：多尺度和细节增强的任意分割模型（A会议 2024）
    讲解视频：https://www.bilibili.com/video/BV1giDmYHEre/
        轻量级多尺度适配器（Lightweight Multi-Scale Adapter，LMSA）：
            通过引入多尺度信息来提高模型的表现，同时采用了轻量级的设计思路，仅需要很少的训练参数即可适应不同的数据集。
'''
class ModifyPPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(ModifyPPM, self).__init__()
        # 初始化特征列表
        self.features = []

        # 定义局部卷积操作
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False, groups=in_dim),  # 深度卷积
            nn.GELU(),  # GELU激活函数
        )

        # --------------关键 多尺度--------------
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),  # 自适应平均池化的作用：将输入特征图缩放到多个不同的尺寸（由 bins 参数指定）
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1),  # 1x1卷积进行通道数缩减
                nn.GELU(),
                nn.Conv2d(reduction_dim, reduction_dim, kernel_size=3, bias=False, groups=reduction_dim),  # 分组卷积
                nn.GELU()
            ))
        # ------------------------------------

        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()  # 获取输入张量的尺寸

        out = [self.local_conv(x)]  # 应用局部卷积并将结果添加到输出列表

        # 对每个特征层进行插值并添加到输出列表
        for f in self.features:
            """
                F.interpolate：将缩放后的特征图通过双线性插值恢复到原始尺寸，以便与其他特征拼接
            """
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))

        return torch.cat(out, 1)  # 将所有输出在通道维度上拼接

class LMSA(nn.Module):
    """
        局部多尺度注意力机制
    """
    #     in_dim = 256
    #     hidden_dim = 128
    def __init__(self, in_dim, hidden_dim, patch_num):
        super().__init__()
        # 定义降维线性层
        self.down_project = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()

        self.mppm = ModifyPPM(hidden_dim, hidden_dim // 4, [3, 6, 9, 12])

        self.patch_num = patch_num  # 存储patch数量

        # 定义升维线性层
        self.up_project = nn.Linear(hidden_dim, in_dim)

        # 定义降维卷积层
        self.down_conv = nn.Sequential(nn.Conv2d(hidden_dim * 2, hidden_dim, 1),nn.GELU())

    def forward(self, x):
        # torch.Size([1, 16, 16, 256])
        down_x = self.down_project(x)  # 应用降维线性层 torch.Size([1, 16, 16, 128])
        down_x = self.act(down_x)  # 应用激活函数 torch.Size([1, 16, 16, 128])

        # 调整张量维度以适应卷积操作
        down_x = down_x.permute(0, 3, 1, 2).contiguous()    # torch.Size([1, 128, 16, 16])
        # 【多尺度特征】
        down_x = self.mppm(down_x).contiguous()  # 通过ModifyPPM模块    torch.Size([1, 256, 16, 16])

        down_x = self.down_conv(down_x)  # 应用降维卷积层  torch.Size([1, 128, 16, 16])
        # 调整张量维度以恢复原始形状
        down_x = down_x.permute(0, 2, 3, 1).contiguous()    # torch.Size([1, 16, 16, 128])

        up_x = self.up_project(down_x)  # 应用升维线性层  torch.Size([1, 16, 16, 256])
        return x + up_x  # 输出为输入和处理后的张量之和


if __name__ == "__main__":
    in_dim = 256
    hidden_dim = 128
    patch_num = 16
    input_tensor = torch.randn(1, patch_num, patch_num, in_dim)

    model = LMSA(in_dim, hidden_dim, patch_num)
    output = model(input_tensor)

    print('Input size:', input_tensor.size())   # Input size: torch.Size([1, 16, 16, 256])
    print('Output size:', output.size())        # Output size: torch.Size([1, 16, 16, 256])

    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息