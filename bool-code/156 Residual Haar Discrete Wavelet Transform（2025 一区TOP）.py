import torch.nn as nn
import torch
from pytorch_wavelets import DWTForward

# 报错则安装：pip install pytorch_wavelets
"""    
    论文地址：https://arxiv.org/pdf/2401.15578
    论文题目：ASCNet: Asymmetric Sampling Correction Network for Infrared Image Destriping （TGRS 2025）
    中文题目：ASCNet：用于红外图像去条带的非对称采样校正网络 （TGRS 2025）
    讲解视频：https://www.bilibili.com/video/BV1bPJEzwE5J/
        残差Haar离散小波变换（Residual Haar Discrete Wavelet Transform，RHDWT）：
        实际意义：①离散小波变换的局限性：仅依赖空间采样和方向先验（如条带噪声的水平梯度特性），但缺乏跨通道的语义交互，导致特征仅包含空间信息，无法捕捉数据驱动的高层语义。
                ②步长卷积：虽能融合空间与全语义特征，但忽略噪声方向先验，导致特征中方向信息缺失。
        实现方式：双分支并行：①模型驱动分支：Haar小波分解，提取条带方向先验。
                           ②残差分支：卷积融合空间与跨通道语义。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class Residual_Haar_Discrete_Wavelet_Transform(nn.Module):
    """残差Haar离散小波变换模块
    通过小波变换实现下采样，同时保留原始特征信息
    参数：
        in_channels：输入特征图的通道数
        n：输出通道的扩展倍数（默认不扩展）
    """
    def __init__(self, in_channels, n=1):
        super(Residual_Haar_Discrete_Wavelet_Transform, self).__init__()
        # 残差路径的3x3卷积（stride=2实现下采样，padding=1保持尺寸对齐）
        self.identety = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * n,  # 输出通道数控制
            kernel_size=3,
            stride=2,  # 下采样2倍
            padding=1
        )

        # 创建Haar小波变换实例（J=1表示单层分解）
        self.DWT = DWTForward(J=1, wave='haar')

        # 小波特征编码模块
        self.dconv_encode = nn.Sequential(
            # 输入通道是4倍因为小波分解产生4个分量
            nn.Conv2d(in_channels * 4, in_channels * n, 3, padding=1),
            nn.LeakyReLU(inplace=True),  # 带泄露的ReLU激活函数
        )

    def _transformer(self, DMT1_yl, DMT1_yh):
        """重组小波分解结果
        将低频分量(DMT1_yl)和三个高频分量(DMT1_yh)拼接

        参数：
            DMT1_yl：低频分量张量 [N, C, H/2, W/2]
            DMT1_yh：高频分量列表 [N, C, 3, H/2, W/2]

        返回：
            拼接后的特征张量 [N, 4*C, H/2, W/2]
        """
        list_tensor = []
        a = DMT1_yh[0]  # 提取高频分量（J=1时只有第一层分解结果）
        list_tensor.append(DMT1_yl)  # 添加低频分量
        for i in range(3):
            # 遍历水平、垂直、对角线三个高频分量
            list_tensor.append(a[:, :, i, :, :])
        return torch.cat(list_tensor, 1)  # 沿通道维度拼接

    def forward(self, x):
        # 保留原始输入用于残差连接
        input = x

        # 执行Haar小波分解
        # DMT1_yl: 低频分量 [N, C, H/2, W/2]
        # DMT1_yh: 高频分量列表（每个元素是[N, C, 3, H/2, W/2]）
        DMT1_yl, DMT1_yh = self.DWT(x)

        # 重组小波分量（通道数变为4倍）
        DMT = self._transformer(DMT1_yl, DMT1_yh)
        # 对重组特征进行编码（通道数调整）
        x = self.dconv_encode(DMT)

        # 残差路径处理（3x3卷积下采样）
        res = self.identety(input)

        # 特征融合（元素级相加）
        out = torch.add(x, res)
        return out
# 下采样
if __name__ == '__main__':
    x = torch.randn(1, 32, 50, 50)  # 输入 [batch, channels, height, width]
    model = Residual_Haar_Discrete_Wavelet_Transform(in_channels=32, n=1)
    output = model(x)
    print(f"输入张量形状: {x.shape}")  # [1, 32, 50, 50]
    print(f"输出张量形状: {output.shape}")  # [1, 32, 25, 25]（空间维度下采样2倍）
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")