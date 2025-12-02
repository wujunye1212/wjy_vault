import torch
import torch.nn as nn
from einops import rearrange
import torch_dct # pip install torch_dct -i https://pypi.tuna.tsinghua.edu.cn/simple
from torch.nn import functional as F

#看Ai缝合怪b站视频：2025.6.13更新的视频

# 定义一个名为 FEFM 的类
class FEFM(nn.Module):
    # 初始化函数，接受输入通道数 in_channels、patch_size（默认为4）、init_lambda（控制权重，默认为0.5）
    def __init__(self, in_channels, patch_size=4, init_lambda=0.5):
        # 调用父类 nn.Module 的初始化函数
        super(FEFM, self).__init__()

        # 定义一个可学习的参数（temperature），用于注意力机制中的缩放因子
        self.temperature = nn.Parameter(torch.ones(1, 1))

        # 定义三个卷积层：
        # conv_Fn_v：用于提取 Value 特征图
        # conv_Fn_k：用于提取 Key 特征图
        # conv_Fr_q：用于提取 Query 特征图
        self.conv_Fn_v = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_Fn_k = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_Fr_q = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # 保存初始化参数 init_lambda 和 patch_size 到类属性中
        self.init_lambda = init_lambda
        self.patch_size = patch_size

    # 前向传播函数，接受两个输入张量 input1 和 input2
    def forward(self, input1, input2):
        # 获取 input1 的形状 (batch_size, channels, height, width)
        b, c, h, w = input1.shape

        # 使用不同的卷积核分别提取 Query、Key、Value 特征图
        F_Q = self.conv_Fr_q(input1)  # Query 特征图
        F_K = self.conv_Fn_k(input1)  # Key 特征图
        F_V = self.conv_Fn_v(input2)  # Value 特征图

        # 将 Query 特征图按照 patch_size 进行分块，便于后续 DCT 变换
        q_patch = rearrange(F_Q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',patch1=self.patch_size, patch2=self.patch_size)

        # 同上，对 Key 特征图进行分块处理
        k_patch = rearrange(F_K, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',patch1=self.patch_size, patch2=self.patch_size)

        # 对分块后的 Query 和 Key 进行二维离散余弦变换（DCT）
        fft_Q = torch_dct.dct_2d(q_patch.float())
        fft_K = torch_dct.dct_2d(k_patch.float())

        # 计算 Query 和 Key 在频域下的点乘（类似注意力机制中的 QK^T）
        F1 = fft_Q @ fft_K  # 等价于 torch.matmul(fft_Q, fft_K)

        # 将结果展平以便后续计算
        F1 = rearrange(F1, 'b c h w patch1 patch2 -> b c (h w patch1 patch2)',patch1=self.patch_size, patch2=self.patch_size)

        # 展平 Query 的频域特征
        Fq = rearrange(fft_Q, 'b c h w patch1 patch2 -> b c (h w patch1 patch2)',patch1=self.patch_size, patch2=self.patch_size)

        # 展平 Key 的频域特征
        Fk = rearrange(fft_K, 'b c h w patch1 patch2 -> b c (h w patch1 patch2)',patch1=self.patch_size, patch2=self.patch_size)

        # 计算注意力矩阵 attn，通过 QK^T 并乘以 temperature 缩放因子
        attn = (Fq @ Fk.transpose(-2, -1)) * self.temperature

        # 使用 softmax 生成注意力矩阵
        attn = attn.softmax(dim=-1)

        # 使用注意力矩阵对 F1 进行加权处理，得到 Fcfr
        Fcfr = attn @ F1

        # 将 Fcfr reshape 成原来的块结构形式
        Fcfr = Fcfr.reshape(b, c, h // self.patch_size, w // self.patch_size,self.patch_size, self.patch_size)

        # 对 Fcfr 进行二维逆离散余弦变换（IDCT）还原到空间域
        Fcfr_out = torch_dct.idct_2d(Fcfr)

        # 将块状结构重新拼接回原始图像大小
        Fcfrout = rearrange(Fcfr_out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size, patch2=self.patch_size)

        # DFR 操作部分
        # 将 Fcfrout 与 Value 相乘（类似于 Transformer 中的 V 加权）
        out = Fcfrout @ F_V

        # 使用 init_lambda 控制残差连接的比例
        Fdfr = torch.sub( F_V,out * self.init_lambda)

        # 最终输出是 Fcfrout 与 Query 的乘积加上 DFR 残差
        out = Fcfrout @ F_Q + Fdfr

        # 返回最终输出张量
        return out

# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input1= torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 32, 64, 64)
    # 创建 FEFM 模块
    fefm = FEFM(in_channels=32)
    # 将输入图像传入 FEFM 模块进行处理
    output = fefm(input1,input2)
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新-FEFM_input_size:', input1.size())
    print('Ai缝合即插即用模块永久更新-FEFM_output_size:', output.size())

    # CVPR2025 FEFM模块的二次创新，CFEM在我的二次创新模块交流群，冲SCI三区和四区，CCF-B/C,可以直接去发小论文！
    # 创建 CFEM 模块
    # cfem = CFEM(in_channels=32)
    # # 将输入图像传入CFEM 模块进行处理
    # output = cfem(input1,input2)
    # print('顶会顶刊二次创新模块永久更新在二次创新交流群-CFEM_input_size:', input1.size())
    # print('顶会顶刊二次创新模块永久更新在二次创新交流群-CFEM_output_size:', output.size())
    #CVPR2025 FEFM模块的二次创新，CFEM在我的二次创新模块改进交流群，可以直接去发小论文！
    #CFEM二次创新模块只更新二次创新交流，永久更新中
    #二次创新改进商品链接在视频评论区


