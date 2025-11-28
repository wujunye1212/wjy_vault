import torch
from torch import nn
'''
ACM MM2024 属于CCF-A类顶会论文
两个即插即用模块：
 MSDE 多尺度细节增强模块
 MEEM 多尺度边缘增强模块 

MEEM多尺度边缘增强模块的作用：
解决显著目标检测（SOD）中细节边缘缺失的问题，提高模型捕捉物体边界的能力。
使用3×3平均池化和1×1卷积从输入图像中提取多尺度的边缘信息。
通过边缘增强器（EE），在每个尺度上强化边缘感知，突出物体的关键边界。
提取的多尺度边缘信息与主分支的特征融合，提升最终预测结果的精细度。

MSDE 多尺度细节增强模块的作用：
补充显著物体中缺乏的细节信息，解决SAM模型在解码过程中的上采样无法恢复关键细节的问题。
包含主分支和辅助分支，主分支负责逐步上采样特征图，辅助分支则从输入图像中提取精细细节信息，并与主分支融合。
通过引入MEEM模块，辅助分支能够在不同尺度上捕捉到边缘细节，并与主分支的特征结合，实现精细化的显著物体检测。
这两个模块相辅相成，使模型能够在复杂场景下准确捕捉目标边缘和细节，大幅提升分割精度和结果的可解释性。
'''
class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),
            norm(in_dim),
            act()
        )

    def forward(self, x):
        mid = self.in_conv(x)

        out = mid
        # print(out.shape)

        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)

        out = self.out_conv(out)

        return out


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class DetailEnhancement(nn.Module):
    def __init__(self, img_dim, feature_dim, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.img_in_conv = nn.Sequential(
            nn.Conv2d(img_dim,feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act()
        )
        self.img_er = MEEM(feature_dim, feature_dim // 2, 4, norm, act)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim *2, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Conv2d(feature_dim, feature_dim * 2, 3, padding=1, bias=False),
            norm(feature_dim * 2),
            act(),
        )

        self.out_conv = nn.Conv2d(feature_dim * 2, img_dim, 1)

        self.feature_upsample = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
        )

    def forward(self, img, feature, b_feature):
        feature = torch.cat([feature, b_feature], dim=1)
        feature = self.feature_upsample(feature)

        img_feature = self.img_in_conv(img)
        img_feature = self.img_er(img_feature) + img_feature

        out_feature = torch.cat([feature, img_feature], dim=1)
        out_feature = self.fusion_conv(out_feature)
        out = self.out_conv(out_feature)
        return out

# 输入 B C H W,  输出B C H W
if __name__ == "__main__":
    # 创建DetailEnhancement模块的实例
    # 第一个即插即用模块是多尺度细节增强模块：MSDE
    MSDE =DetailEnhancement(img_dim=64,feature_dim=128)
    input = torch.randn(1, 64, 128, 128)
    feature = torch.randn(1, 128, 32, 32)
    b_feature = torch.randn(1, 128, 32, 32)
    # 执行前向传播
    output= MSDE(input,feature,b_feature)
    print('MSDE Input size:', input.size())
    print('MSDE Output size:', output.size())
    print('---------------------------------')

    # 第二个即插即用模块是多尺度边缘增强模块：MEEM
    MEEM = MEEM(in_dim=64,hidden_dim=32)
    input = torch.randn(1, 64, 128, 128)
    # 执行前向传播
    output = MEEM(input)
    print('MEEM Input size:', input.size())
    print('MEEM Output size:', output.size())