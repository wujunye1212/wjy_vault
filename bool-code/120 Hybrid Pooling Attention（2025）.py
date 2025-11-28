import torch
import torch.nn as nn

"""
    论文地址：https://www.sciencedirect.com/science/article/pii/S1051200425000922
    论文题目：A synergistic CNN-transformer network with pooling attention fusion for hyperspectral image classification（2025）
    中文题目：一种用于高光谱图像分类的融合池化注意力机制的协同卷积神经网络（2025）
    讲解视频：https://www.bilibili.com/video/BV1QBP3enEoY/
        混合池化注意力（Hybrid Pooling Attention, HPA）：
            实际意义：①空间信息利用不足：没有充分利用高光谱图像的丰富空间信息。
                    ②特征融合不佳：没有得到有效的结合不同的池化操作。
                    ③特征表示有限：传统方法无法充分体现不同通道和空间位置之间的关系。
            实现方式：首先，输入特征图M沿通道维度分成G个子特征。然后通过两个分支沿高度和宽度维度进行全局平均池化/最大池化,
                    实现近似窗口内值/捕获窗口内峰值作用，再经Sigmoid函数来重新校准通道间关系。然后经Group Norm层增强稳定性，
                    再分别用2D全局平均池化和最大池化编码空间信息，通过输出矩阵点乘得到最终特征图。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

# Hybrid Pooling Attention
class HPA(nn.Module):
    def __init__(self, channels,factor=32):
        """混合池化注意力模块
        Args:
            channels: 输入通道数
            factor: 分组数，默认32组
        """
        super(HPA, self).__init__()
        # 基础参数校验
        self.groups = factor  # 将通道分成多少组处理（类似分组卷积）
        assert channels // self.groups > 0  # 确保每组至少有1个通道

        # ----------------- 注意力机制组件 -----------------
        # 双池化分支：同时利用平均池化和最大池化捕捉不同特征
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化（输出1x1）
        self.map = nn.AdaptiveMaxPool2d((1, 1))  # 全局最大池化（输出1x1）

        # 空间维度池化（分别提取高度/宽度方向特征）
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 高度方向平均池化（保持宽度维度）
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 宽度方向平均池化（保持高度维度）
        self.max_h = nn.AdaptiveMaxPool2d((None, 1))  # 高度方向最大池化
        self.max_w = nn.AdaptiveMaxPool2d((1, None))  # 宽度方向最大池化

        # ----------------- 特征变换层 -----------------
        self.gn = nn.GroupNorm(
            num_groups=channels // self.groups,  # 分组数=每组通道数
            num_channels=channels // self.groups  # 每组输入通道数
        )
        self.conv1x1 = nn.Conv2d(
            in_channels=channels // self.groups,
            out_channels=channels // self.groups,
            kernel_size=1,  # 用于通道间信息融合
            stride=1, padding=0
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=channels // self.groups,
            out_channels=channels // self.groups,
            kernel_size=3,  # 捕捉局部空间特征
            padding=1  # 保持特征图尺寸不变
        )
        self.softmax = nn.Softmax(dim=-1)  # 用于注意力权重归一化

    def forward(self, x):
        # 输入x形状: [batch_size, channels, height, width]
        b, c, h, w = x.size()

        # ============= 特征分组处理 =============
        # 将通道维度拆分为groups组：[b,c,h,w] -> [b*groups, c/groups, h,w]
        group_x = x.reshape(b * self.groups, -1, h, w)  # -1自动计算为c/groups

        # ============= 平均池化分支 =============
        # 沿高度和宽度方向分别池化
        x_h = self.pool_h(group_x)  # 形状: [b*g, c/g, h, 1]
        x_w = self.pool_w(group_x)  # 形状: [b*g, c/g, 1, w]
        x_w = x_w.permute(0, 1, 3, 2)  # 维度置换：[b*g, c/g, w, 1]
        # 拼接后通过1x1卷积融合空间信息
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # dim=2在高度维度拼接
        # 拆分回原始维度（利用切片操作）
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # x_h形状恢复为[b*g, c/g, h,1]
        #------------ 到此，是为了得到H 和 W维度上 平均池化的权重参数 ------------
        x1 = self.gn(  # 分组归一化增强训练稳定性
            group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        )

        # ============= 最大池化分支 =============
        # 处理逻辑与平均池化分支类似，但使用最大池化
        y_h = self.max_h(group_x)  # 形状: [b*g, c/g, h, 1]
        y_w = self.max_w(group_x).permute(0, 1, 3, 2)  # 维度置换
        yhw = self.conv1x1(torch.cat([y_h, y_w], dim=2))
        y_h, y_w = torch.split(yhw, [h, w], dim=2)
        #------------ 到此，是为了得到H 和 W维度上 最大池化的权重参数 ------------
        y1 = self.gn(
            group_x * y_h.sigmoid() * y_w.permute(0, 1, 3, 2).sigmoid()
        )

        # ============= 注意力权重融合 =============
        # 处理平均池化分支
        x11 = x1.reshape(b * self.groups, -1, h * w)  # 展平空间维度：[b*g, c/g, h*w]
        x12 = self.agp(x1)  # 全局平均池化：[b*g, c/g, 1, 1]
        x12 = x12.reshape(b * self.groups, -1, 1)  # [b*g, c/g, 1]
        x12 = x12.permute(0, 2, 1)  # 调整为矩阵乘法维度：[b*g, 1, c/g]
        x12 = self.softmax(x12)  # 归一化得到注意力权重

        # 处理最大池化分支
        y11 = y1.reshape(b * self.groups, -1, h * w)  # [b*g, c/g, h*w]
        y12 = self.map(y1)  # 全局最大池化
        y12 = y12.reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        y12 = self.softmax(y12)

        # 双分支权重融合（矩阵乘法实现跨通道交互）
        weights = (
                torch.matmul(x12, y11) +  # [b*g, 1, h*w]
                torch.matmul(y12, x11)  # [b*g, 1, h*w]
        ).reshape(b * self.groups, 1, h, w)  # 恢复空间维度

        # ============= 最终输出 =============
        # 应用Sigmoid激活并还原分组前的维度
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

if __name__ == "__main__":
    # 测试代码
    model = HPA(64)  # 实例化模块（输入64通道）
    input = torch.randn(2, 64, 128, 128)  # 模拟输入：batch_size=2
    output = model(input)  # 前向传播

    print('input_size:', input.size())  # 打印输入尺寸
    print('output_size:', output.size())  # 打印输出尺寸

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')
    print("公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")