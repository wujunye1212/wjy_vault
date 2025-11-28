import torch
import torch.nn as nn
'''
    论文地址：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05751.pdf
    论文题目：Efficient Frequency-Domain Image Deraining with Contrastive Regularization （ECCV 2024）
    中文题目： 利用对比正则化实现高效的频域图像去雨（ECCV 2024）
    讲解视频：https://www.bilibili.com/video/BV15BUzYcEbi/
        融合傅里叶卷积混合器（Fused Fourier Convolution Mixer,FFCM）：
               依据：空间-频率的混合特征提取
               优点：具有与自注意力机制相似的全局感受野，但计算成本与卷积相同。
'''
class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        # 定义卷积层，用于处理傅里叶变换后的特征
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        # 定义批归一化层
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        # 定义ReLU激活函数
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        # 进行二维傅里叶变换
        ffted = torch.fft.rfft2(x, norm='ortho')
        # 获取傅里叶变换的实部
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        # 获取傅里叶变换的虚部
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        # 合并实部和虚部
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # 调整维度顺序
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        # 调整形状以适应卷积层输入
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # 通过卷积层
        ffted = self.conv_layer(ffted)
        # 通过批归一化和ReLU激活
        ffted = self.relu(self.bn(ffted))

        # 调整形状以适应傅里叶逆变换
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        # 转换为复数形式
        ffted = torch.view_as_complex(ffted)

        # 进行逆傅里叶变换
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class Freq_Fusion(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size=[1,3,5,7],
            se_ratio=4,
            local_size=8,
            scale_ratio=2,
            spilt_num=4
    ):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim*scale_ratio//spilt_num

        # 定义初始卷积层1
        self.conv_init_1 = nn.Sequential(  # PW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        # 定义初始卷积层2
        self.conv_init_2 = nn.Sequential(  # DW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        # 定义中间卷积层
        self.conv_mid = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        # 定义傅里叶单元
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

        # 定义批归一化和ReLU激活
        self.bn = torch.nn.BatchNorm2d(dim*2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        # 将输入张量分割为两个
        x_1, x_2 = torch.split(x, self.dim, dim=1)

        # 通过初始卷积层1
        x_1 = self.conv_init_1(x_1)
        # 通过初始卷积层2
        x_2 = self.conv_init_2(x_2)

        # 合并两个特征图
        x0 = torch.cat([x_1, x_2], dim=1)
        # 通过傅里叶单元并进行残差连接
        x = self.FFC(x0) + x0
        # 通过批归一化和ReLU激活
        x = self.relu(self.bn(x))

        return x


class FFCM(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_gloal=Freq_Fusion,
            mixer_kernel_size=[1,3,5,7],
            local_size=8
    ):
        super(FFCM, self).__init__()
        self.dim = dim
        # 定义全局特征融合模块
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim, kernel_size=mixer_kernel_size,
                                 se_ratio=8, local_size=local_size)
        # 定义通道注意力卷积层
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        # 定义通道注意力模块
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        # 定义初始卷积层
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        # 定义深度可分离卷积层1
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=3 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )
        # 定义深度可分离卷积层2
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=5, padding=5 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )

    def forward(self, x):
        # 通过初始卷积层
        x = self.conv_init(x)
        # 将特征图分割
        x = list(torch.split(x, self.dim, dim=1))
        # 通过深度可分离卷积层1
        x_local_1 = self.dw_conv_1(x[0])
        # 通过深度可分离卷积层2
        x_local_2 = self.dw_conv_2(x[0])

        # 通过全局特征融合模块
        x_gloal = self.mixer_gloal(torch.cat([x_local_1, x_local_2], dim=1))

        # 通过通道注意力卷积层
        x = self.ca_conv(x_gloal)
        # 应用通道注意力
        x = self.ca(x) * x
        return x

if __name__ == '__main__':

    ffcm_module = FFCM(dim=32)

    input = torch.randn(1, 32, 128, 128)
    # 将输入张量传入 FFCM 模块
    output = ffcm_module(input)
    # 输出结果的形状
    print("输入张量的形状：", input.shape)
    print("输出张量的形状：", output.shape)

    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息
