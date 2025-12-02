import torch
import torch.nn as nn
from einops import rearrange
"""
论文地址：https://ieeexplore.ieee.org/abstract/document/10445289/
论文题目：Hybrid Convolutional and Attention Network for Hyperspectral Image Denoising (2024)
中文题目：用于高光谱图像去噪的混合卷积和注意力网络
讲解视频：https://www.bilibili.com/video/BV1tT2VYzETA/
        Convolution and Attention Fusion Module
        卷积和注意力融合模块：
        作用：捕获长距离依赖关系和邻域光谱相关性，能够有效地融合全局和局部信息，从而提高去噪效果。
        原理：卷积操作由于其局部性质和受限感知范围，不足以建模全局特征。Transformer通过注意力机制在提取全局特征和捕获长距离依赖方面表现出色。
                    卷积和注意力是互补的，可以同时建模全局和局部特征。
                    
                    可用于改进：https://www.bilibili.com/video/BV1Cv2WYFEnH/
"""
class CAFM(nn.Module):
    # 定义构造函数，初始化模型参数
    def __init__(self, dim, num_heads=4, bias=False):
        super(CAFM, self).__init__()  # 调用父类的构造函数
        self.num_heads = num_heads  # 设置头的数量
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 初始化超参数

        # 定义查询、键和值的卷积层
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        # 深度可分离卷积层，用于qkv
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,bias=bias)

        # 输出投影层
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)

        # 全连接层，用于调整通道数
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        # 深度卷积层，用于局部特征提取
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,groups=dim // self.num_heads, padding=1)

    # 前向传播函数
    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入张量的尺寸 torch.Size([1, 64, 32, 32])

        x = x.unsqueeze(2)  # 在第2维度增加一个维度，以便进行3D卷积  torch.Size([1, 64, 1, 32, 32])
        qkv = self.qkv_dwconv(self.qkv(x))  # 对输入执行QKV卷积，并通过深度可分离卷积
        qkv = qkv.squeeze(2)  # 移除之前添加的维度   torch.Size([1, 64, 32, 32])

        # ======================局部特征======================
        f_conv = qkv.permute(0, 2, 3, 1)  # 改变维度顺序以适应后续操作
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)  # 重塑并转置张量
        f_all = self.fc(f_all.unsqueeze(2))  # 执行全连接层操作
        f_all = f_all.squeeze(2)  # 再次移除维度

        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)  # 重塑张量为适合局部卷积的形状
        f_conv = f_conv.unsqueeze(2)  # 添加维度
        out_conv = self.dep_conv(f_conv)  # 执行局部卷积
        out_conv = out_conv.squeeze(2)  # 移除维度

        # ======================全局特征======================
        q, k, v = qkv.chunk(3, dim=1)  # 将qkv分割成三个部分：查询、键和值
        # 重新排列张量，使得每个头可以独立地计算注意力
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # 归一化查询和键
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)  # 使用softmax归一化注意力权重
        # 应用注意力到值上
        out = (attn @ v)
        # 重新排列输出张量
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)  # 添加维度
        out = self.project_out(out)  # 投影输出
        out = out.squeeze(2)  # 移除维度

        output = out + out_conv  # 结合全局自注意力结果与局部卷积结果
        return output  # 返回最终输出

if __name__ == '__main__':
    input = torch.rand(1, 64, 32, 32)  # 创建随机输入张量

    CAFM = CAFM(dim=64)

    output = CAFM(input)

    print('input_size:', input.size())
    print('output_size:', output.size())

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")