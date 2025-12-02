import math
import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2307.14010
    论文题目：ESSAformer: Efﬁcient Transformer for Hyperspectral Image Super-resolution (ECCV 2023)
    中文题目：MSA2Net：用于医学图像分割的多尺度自适应注意力引导网络
    讲解视频：https://www.bilibili.com/video/BV15n61YtEZX/
        高效SCC自注意力（Efficient SCC-kernel-based self-attention,ESSA）
             理论研究：通过找到映射函数将自注意力转换为线性复杂度，从而降低了计算负担。
"""
class ESSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 创建线性层，用于Q、K、V的转换，输出维度是输入的3倍
        self.lnqkv = nn.Linear(dim, dim * 3)
        # 创建输出的线性层，维度保持不变
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        # 获取输入张量的维度信息：批次、通道数、高度、宽度
        b,c,h,w = x.shape
        # 重塑张量并调整维度顺序，准备进行注意力计算
        x = x.reshape(b,c,h*w).permute(0,2,1)
        b, N, C = x.shape
        # 通过线性层生成Q、K、V
        qkv = self.lnqkv(x)
        # 将结果分割成三份，对应Q、K、V
        qkv = torch.split(qkv, C, 2)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # 大佬解析：https://zhuanlan.zhihu.com/p/717463998
        # 计算并移除Q的均值（中心化）
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        # 计算并移除K的均值（中心化）
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        # 计算Q的平方
        q2 = torch.pow(q, 2)
        # 计算Q平方的和
        q2s = torch.sum(q2, dim=2, keepdim=True)
        # 计算K的平方
        k2 = torch.pow(k, 2)
        # 计算K平方的和
        k2s = torch.sum(k2, dim=2, keepdim=True)

        # 第一项：原始值V
        t1 = v

        # 对K进行归一化
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        # 对Q进行归一化
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        # 计算注意力得分并与V相乘，除以sqrt(N)进行缩放
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)

        # 将两部分相加得到最终的注意力输出
        attn = t1 + t2
        # 通过线性层处理
        attn = self.ln(attn)
        # 重塑张量回原始维度顺序
        x = attn.reshape(b,h,w,c).permute(0,3,1,2)
        return x

if __name__ == "__main__":
    # 创建测试输入张量
    input =  torch.randn(1, 32, 77, 77)
    # 实例化模型
    model = ESSA(32)
    # 进行前向传播
    output = model(input)
    # 打印输入输出的尺寸
    print('input_size:',input.size())
    print('output_size:',output.size())
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")