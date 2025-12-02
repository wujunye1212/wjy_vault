import torch
from torch import nn
import torch.nn.functional as F

"""
    论文地址：https://ieeexplore.ieee.org/document/10841446
    论文题目：An Adaptive Dual-Supervised Cross-Deep Dependency Network for Pixel-Wise Classiﬁcation（TGRS 2025）
    中文题目：用于像素级分类的自适应双监督交叉深度依赖网络（TGRS 2025）
    讲解视频：https://www.bilibili.com/video/BV1azEWzUEgT/
    可变形交互注意力（Deformable Interactive Attention Module，DIAM）：
        实际意义：①图像离散性的采样难题：图像离散特性导致采样点难以处理，阻碍空间信息收敛。
                ②注意力机制的局限性：传统注意力机制易聚焦优势特征（多），这种偏向会造成其他特征丢失，限制不同特征间的交互增强。    
        实现方式：①DIAM通过弱化不利于提升的样本点，优化模型对空间特征的捕捉能力。
                ②DIA-Module利用不同池化策略生成具有属性差异特征，充分挖掘不同属性特征的潜力，促进神经元和可学习参数在训练中更积极地参与，增强掩码空间特征的表示能力，提升各类特征的综合利用效率。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class DeformableInteractiveAttention(nn.Module):
    def __init__(self, stride=1, distortionmode=False):
        super(DeformableInteractiveAttention, self).__init__()

        # 特征融合卷积层（2通道输入 -> 1通道输出）
        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

        # 非线性激活函数
        self.sigmoid = nn.Sigmoid()

        # 模式开关
        self.distortionmode = distortionmode

        # 上采样层（用于恢复特征图分辨率）
        self.upsample = nn.Upsample(scale_factor=2)

        # 两种下采样方式：平均池化和最大池化的替代实现
        self.downavg = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)  # 模拟平均下采样
        self.downmax = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)  # 模拟最大下采样

        # 形变调制模式专用层
        if distortionmode:
            # 第一个调制卷积（处理平均特征）
            self.d_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv.weight, 0)  # 权重初始化为零
            # 注册反向传播钩子（控制梯度回传）
            self.d_conv.register_full_backward_hook(self._set_lra)

            # 第二个调制卷积（处理最大特征）
            self.d_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv1.weight, 0)
            self.d_conv1.register_full_backward_hook(self._set_lrm)

    @staticmethod
    def _set_lra(module, grad_input, grad_output):
        """调制卷积A的学习率调整钩子（减少梯度更新量）"""
        # 将输入和输出梯度乘以0.4
        grad_input = [g * 0.4 if g is not None else None for g in grad_input]
        grad_output = [g * 0.4 if g is not None else None for g in grad_output]
        return tuple(grad_input), tuple(grad_output)

    @staticmethod
    def _set_lrm(module, grad_input, grad_output):
        """调制卷积M的学习率调整钩子（更小的更新量）"""
        # 将输入和输出梯度乘以0.1
        grad_input = [g * 0.1 if g is not None else None for g in grad_input]
        grad_output = [g * 0.1 if g is not None else None for g in grad_output]
        return tuple(grad_input), tuple(grad_output)

    def forward(self, x):
        # 通道维度压缩（平均池化和最大池化）
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B,1,H,W]

        # 下采样操作（缩小分辨率）
        avg_out = self.downavg(avg_out)
        max_out = self.downmax(max_out)

        # 形变调制处理
        if self.distortionmode:
            # 对下采样特征进行调制
            d_avg_out = torch.sigmoid(self.d_conv(avg_out))  # 平均特征调制
            d_max_out = torch.sigmoid(self.d_conv1(max_out))  # 最大特征调制
            # 交叉调制融合
            max_out = d_avg_out * max_out  # 用平均特征调制最大特征
            avg_out = d_max_out * avg_out  # 用最大特征调制平均特征

        # 拼接特征图（通道维度）
        out = torch.cat([max_out, avg_out], dim=1)  # [B,2,H/2,W/2]
        # 特征融合
        out = self.conv(out)  # [B,1,H/2,W/2]
        # 生成注意力掩码（上采样恢复分辨率）
        mask = self.sigmoid(self.upsample(out))  # [B,1,H,W]

        # 应用注意力机制
        att_out = x * mask  # 特征图加权
        return F.relu(att_out)

if __name__ == '__main__':
    x = torch.randn(1, 32, 50, 50)
    model = DeformableInteractiveAttention(stride=1, distortionmode=True)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")