from einops import rearrange
import torch
from torch.nn import functional as F
from torch import nn
'''
    论文地址：https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_DNF_Decouple_and_Feedback_Network_for_Seeing_in_the_Dark_CVPR_2023_paper.pdf
    论文题目：DNF: Decouple and Feedback Network for Seeing in the Dark （CVPR 2023）
    中文题目：DNF：在黑暗中看到的解耦和反馈
    讲解视频：https://www.bilibili.com/video/BV14PUGYzE4M/
        矩阵化颜色校正（Matrixed Color Correction，MCC）：
             启发点：图像颜色主要通过逐通道矩阵变换来增强或转换到其他颜色空间。
             作用  ：用于执行全局颜色增强和局部细化。
'''
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        # 初始化 LayerNorm 类，normalized_shape 表示归一化的维度，eps 是防止除零的小常数
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 可学习的权重参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))   # 可学习的偏置参数
        self.eps = eps  # 设置 epsilon 值
        self.data_format = data_format  # 数据格式，默认为 "channels_last"
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError  # 如果数据格式不支持，抛出异常
        self.normalized_shape = (normalized_shape, )  # 归一化的形状

    def forward(self, x):
        # 前向传播函数
        if self.data_format == "channels_last":
            # 如果是 channels_last 格式，使用内置的 layer_norm
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 如果是 channels_first 格式，手动计算归一化
            u = x.mean(1, keepdim=True)  # 计算均值
            s = (x - u).pow(2).mean(1, keepdim=True)  # 计算方差
            x = (x - u) / torch.sqrt(s + self.eps)  # 归一化
            x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 应用权重和偏置
            return x

class MCC(nn.Module):
    def __init__(self, f_number, num_heads, padding_mode, bias=False) -> None:
        # 初始化 MCC 类，f_number 表示通道数，num_heads 表示头数，padding_mode 表示填充模式
        super().__init__()
        self.norm = LayerNorm(f_number, eps=1e-6, data_format='channels_first')  # 定义 LayerNorm 层

        self.num_heads = num_heads  # 保存头数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 定义可学习的温度参数
        self.pwconv = nn.Conv2d(f_number, f_number * 3, kernel_size=1, bias=bias)  # 逐点卷积，扩展通道数
        self.dwconv = nn.Conv2d(f_number * 3, f_number * 3, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=f_number * 3)  # 深度卷积
        self.project_out = nn.Conv2d(f_number, f_number, kernel_size=1, bias=bias)  # 输出投影层
        self.feedforward = nn.Sequential(
            nn.Conv2d(f_number, f_number, 1, 1, 0, bias=bias),  # 逐点卷积
            nn.GELU(),  # GELU 激活函数
            nn.Conv2d(f_number, f_number, 3, 1, 1, bias=bias, groups=f_number, padding_mode=padding_mode),  # 深度卷积
            nn.GELU()  # GELU 激活函数
        )

    def forward(self, x):
        """
            改进：https://space.bilibili.com/346680886/search/video?keyword=%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B
        """
        attn = self.norm(x)  # 应用 LayerNorm
        _, _, h, w = attn.shape  # 获取输入的高度和宽度

        qkv = self.dwconv(self.pwconv(attn))  # 计算 Q, K, V
        q, k, v = qkv.chunk(3, dim=1)  # 将结果分为 q, k, v 三部分

        # 重塑 q, k, v 以适应多头机制
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)  # 对 q 进行归一化
        k = torch.nn.functional.normalize(k, dim=-1)  # 对 k 进行归一化
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 计算注意力分数
        attn = attn.softmax(dim=-1)  # 对注意力分数进行 softmax

        out = (attn @ v)  # 计算注意力输出

        # 重塑输出
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)  # 应用输出投影
        out = self.feedforward(out + x)  # 应用前馈网络并加上残差连接
        return out

if __name__ == '__main__':
    # 主函数，测试 MCC 模型
    batch_size = 8
    channels = 64
    height = 32
    width = 32
    input = torch.randn(batch_size, channels, height, width)  # 生成随机输入张量

    num_heads = 8
    padding_mode = 'zeros'  # 选择合适的填充模式

    mcc = MCC(f_number=channels, num_heads=num_heads, padding_mode=padding_mode)  # 创建 MCC 模型实例

    output = mcc(input)  # 通过模型计算输出

    print(f"input shape: {input.shape}")  # 打印输入张量的形状
    print(f"output shape: {output.shape}")  # 打印输出张量的形状


    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息
