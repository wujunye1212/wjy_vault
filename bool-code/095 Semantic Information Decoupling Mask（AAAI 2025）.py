import torch.nn as nn
import torch

"""
    论文地址：https://arxiv.org/pdf/2412.08345
    论文题目：ConDSeg: A General Medical Image Segmentation Framework via Contrast-Driven Feature Enhancement (AAAI 2025)
    中文题目：A2RNet：具有对抗攻击鲁棒性的红外和可见光图像融合网络 (AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV1AsrTYNEzc/
        语义信息解耦（Semantic Information Decoupling, SID）
             理论研究：解决边界模糊问题，将高层特征图分解为前景、背景和不确定性区域三个特征图。
"""

# 定义卷积、批归一化和激活函数的模块
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act  # 是否使用激活函数

        # 定义卷积和批归一化的顺序
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)  # 定义ReLU激活函数

    def forward(self, x):
        x = self.conv(x)  # 执行卷积和批归一化
        if self.act == True:  # 如果需要激活函数
            x = self.relu(x)  # 执行ReLU激活
        return x  # 返回结果

# 定义解耦层
class Semantic_Information_Decoupling(nn.Module):
    def __init__(self, in_c=1024, out_c=256):
        super(Semantic_Information_Decoupling, self).__init__()
        # 定义前景特征提取模块
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        # 定义背景特征提取模块
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        # 定义不确定性特征提取模块
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

        # 定义前景分支
        self.branch_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()  # 使用Sigmoid激活
        )
        # 定义背景分支
        self.branch_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()  # 使用Sigmoid激活
        )
        # 定义不确定性分支
        self.branch_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()  # 使用Sigmoid激活
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)  # 前景特征提取
        f_bg = self.cbr_bg(x)  # 背景特征提取
        f_uc = self.cbr_uc(x)  # 不确定性特征提取
        # return f_fg, f_bg, f_uc  # 返回三个特征

        # 图中 Auxiliary Head
        mask_fg = self.branch_fg(f_fg)  # 前景掩码生成
        mask_bg = self.branch_bg(f_bg)  # 背景掩码生成
        mask_uc = self.branch_uc(f_uc)  # 不确定性掩码生成
        return mask_fg, mask_bg, mask_uc  # 返回三个掩码

if __name__ == '__main__':
    input = torch.rand(1, 64, 32, 32)  # 创建一个随机输入张量
    SID = Semantic_Information_Decoupling(64, 64)  # 实例化解耦层模块
    output1, output2, output3 = SID(input)  # 获取输出
    print("SID_input.shape:", input.shape)  # 打印输入形状
    print("SID_output.shape:", output1.shape)  # 打印输出形状
    print("SID_output.shape:", output2.shape)  # 打印输出形状
    print("SID_output.shape:", output3.shape)  # 打印输出形状
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
