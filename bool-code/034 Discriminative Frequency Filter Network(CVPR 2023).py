import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

'''
    论文地址：https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf
    论文题目：Effcient Frequence Domain-based Transformer for High-Quality Image Deblurring（CVPR 2023）
    中文题目：基于高效频域的高质量图像去模糊 Transformers
    讲解视频：https://www.bilibili.com/video/BV1tq1cYiEtD/
        判别式频率域前馈网络 Discriminative Frequency Filter Network（DFFN）：
            自适应地选择重要的频率信息，可用于区分哪些低频和高频信息应该被保留以恢复清晰的图像，提高了特征表示能力。
'''
# 定义一个名为Discriminative_Frequency_Filter_Network的类，该类继承自nn.Module。
class Discriminative_Frequency_Filter_Network(nn.Module):
    # 初始化函数，设置网络层参数。
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        # 调用父类nn.Module的初始化方法。
        super(Discriminative_Frequency_Filter_Network, self).__init__()

        # 根据输入维度dim和扩张因子ffn_expansion_factor计算隐藏特征的数量。
        hidden_features = int(dim * ffn_expansion_factor)

        # 设置patch大小为8x8。
        self.patch_size = 8

        # 存储原始输入维度。
        self.dim = dim
        # 定义第一个卷积层，用于将输入映射到更高维度的空间。
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 定义深度可分离卷积层，保持空间信息的同时减少计算量。
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,groups=hidden_features * 2, bias=bias)

        # 定义一个可学习参数fft，用于频率域滤波。
        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        # 定义最后一个卷积层，将处理后的特征映射回原始维度。
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    # 前向传播函数，定义了数据流经网络的方式。
    def forward(self, x):
        # input_tensor = torch.randn(1, 64, 32, 32)
        # 将输入通过第一个卷积层进行升维。
        x = self.project_in(x)  # torch.Size([1, 340, 32, 32])
        # 重排张量以适应patch尺寸，并分割成多个小块。
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
                            patch1=self.patch_size,patch2=self.patch_size)  # torch.Size([1, 340, 4, 4, 8, 8])

        # 对每个patch执行二维实数傅里叶变换。
        x_patch_fft = torch.fft.rfft2(x_patch.float())      # torch.Size([1, 340, 4, 4, 8, 5])

        # 应用可学习的频率滤波器。
        # x_patch_fft ：torch.Size([1, 340, 4, 4, 8, 5])
        # self.fft:        torch.Size([340, 1, 1, 8, 5])  ---> （广播后）：[1, 340, 4, 4, 8, 5]
        """
            1、最后两个维度都是 [8, 5]，匹配。
            2、我们遇到第一个张量中的维度 4 和第二个张量中的维度 1，这里可以使用广播规则，因为单例维度 1 可以扩展成任何大小以匹配另一个张量的对应维度。
            3、接下来是第一个张量中的又一个维度 4 和第二个张量中的 1，同样可以通过广播来匹配。
            4、对于维度 340，两个张量都有这个维度，并且它的大小相同，所以直接匹配。
            5、在最左边，第一个张量有一个额外的单例维度 1，而第二个张量没有对应的维度，但是由于广播的存在，这不会造成问题。
        """
        x_patch_fft = x_patch_fft * self.fft            # torch.Size([1, 340, 4, 4, 8, 5])

        # 执行逆傅里叶变换，将频域信号转换回时域。
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))       # torch.Size([1, 340, 4, 4, 8, 8])
        # 重新组合处理过的patches成完整的图像。
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
                      patch1=self.patch_size,patch2=self.patch_size)                    # torch.Size([1, 340, 32, 32])

        # 使用深度可分离卷积后，将输出沿通道维度分为两部分。
        """
            x1 torch.Size([1, 170, 32, 32])
            x2 torch.Size([1, 170, 32, 32])
        """
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        # 使用GELU激活函数对第一部分进行非线性变换，然后与第二部分相乘。
        x = F.gelu(x1) * x2
        # 最终通过一个投影层回到原始输入维度。
        x = self.project_out(x)     # torch.Size([1, 64, 32, 32])
        # 返回处理后的输出。
        return x


# 当脚本直接运行时执行以下代码。
if __name__ == '__main__':
    # 创建模型实例，指定输入维度为64。
    model = Discriminative_Frequency_Filter_Network(64)
    # 生成随机输入张量。
    input_tensor = torch.randn(1, 64, 32, 32)

    # 通过模型前向传递得到输出张量。
    output_tensor = model(input_tensor)

    # 打印输入和输出张量的形状。
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")