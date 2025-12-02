import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

"""
论文地址：https://ieeexplore.ieee.org/abstract/document/10445289/
论文题目：Restoring Images in Adverse Weather Conditions via Histogram Transformer (ECCV 2024)
中文题目：Histoformer：基于直方图 Transformer的恶劣天气条件图像恢复
讲解视频：https://www.bilibili.com/video/BV1YGmEYeE27/
        Dynamic-range Histogram Self-Attention
        卷积和注意力融合模块：
        作用：捕获长距离依赖关系和邻域光谱相关性，能够有效地融合全局和局部信息，从而提高去噪效果。
        原理：卷积操作由于其局部性质和受限感知范围，不足以建模全局特征。Transformer通过注意力机制在提取全局特征和捕获长距离依赖方面表现出色。
                    卷积和注意力是互补的，可以同时建模全局和局部特征。

"""
class Dynamic_range_Histogram_SelfAttention(nn.Module):
    # 初始化函数，定义网络结构和参数
    def __init__(self, dim, num_heads=4, bias=False, ifBox=True):
        super(Dynamic_range_Histogram_SelfAttention, self).__init__()  # 调用父类nn.Module的构造函数
        self.factor = num_heads  # 注意力头的数量作为因子
        self.ifBox = ifBox  # 是否使用框处理标志
        self.num_heads = num_heads  # 设置注意力头的数量
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 温度参数，用于缩放点积注意力

        # 使用卷积层将输入维度转换为dim*5
        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        # 深度可分离卷积，不改变通道数但进行局部特征提取
        self.qkv_dwconv = nn.Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)

        # 输出投影层，将结果映射回原始维度
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    # 填充函数，确保输入尺寸可以被factor整除
    def pad(self, x, factor):
        hw = x.shape[-1]  # 获取输入最后一个维度的大小
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]  # 计算需要填充的长度
        x = F.pad(x, t_pad, 'constant', 0)  # 对x进行填充
        return x, t_pad  # 返回填充后的张量和填充信息

    # 移除填充函数
    def unpad(self, x, t_pad):
        _, _, hw = x.shape  # 获取x的形状
        return x[:, :, t_pad[0]:hw - t_pad[1]]  # 根据填充信息移除填充

    # 自定义softmax实现
    def softmax_1(self, x, dim=-1):
        logit = x.exp()  # 对x求指数
        logit = logit / (logit.sum(dim, keepdim=True) + 1)  # 归一化
        return logit  # 返回归一化后的值

    # 归一化函数
    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)  # 计算均值
        sigma = x.var(-2, keepdim=True, unbiased=False)  # 计算方差
        return (x - mu) / torch.sqrt(sigma + 1e-5)  # 归一化并返回

    # 重构注意力机制
    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]  # 获取批次大小和通道数
        q, t_pad = self.pad(q, self.factor)  # 填充q
        k, t_pad = self.pad(k, self.factor)  # 填充k
        v, t_pad = self.pad(v, self.factor)  # 填充v
        hw = q.shape[-1] // self.factor  # 计算每个注意力头的宽度

        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"  # 原始形状模式
        """
        布尔说人话：
            当 ifBox=True 时，通过将空间维度分成块，可以更好地捕捉局部信息。这种做法可能有助于提高模型对局部特征的关注，从而可能减少计算量,更适合那些需要关注局部细节的任务，比如图像分割或目标检测。
            当 ifBox=False 时，数据保持了全局的空间结构，这样可以更好地捕捉长距离依赖关系，这可能会增加计算复杂度,适合需要考虑全局上下文的任务，比如图像分类或语义理解。
        """
        shape_tar = "b head (c factor) hw"  # 目标形状模式

        # 重排qkv以适应注意力计算
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)  # 归一化q
        k = torch.nn.functional.normalize(k, dim=-1)  # 归一化k
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 点积注意力
        attn = self.softmax_1(attn, dim=-1)  # 应用softmax
        out = (attn @ v)  # 计算加权和

        # 重新排列输出到原始形状
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)  # 移除填充
        return out  # 返回注意力机制的结果

    # 前向传播函数
    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入张量的形状

        # 对x的一半进行排序，并记录排序索引 B C H W
        x_sort, idx_h = x[:, :c // 2].sort(-2)  # 表示取 x 的前半部分通道，沿着倒数第二个维度进行排序（H）。idx_h为第一次H排列结果
        x_sort, idx_w = x_sort.sort(-1)     # 这里再次对 x_sort 沿着最后一个维度（W）进行排序。idx_w 为第二次H排列结果
        x[:, :c // 2] = x_sort              # 对输入张量 x 的前半部分通道的数据进行重新排列，使其按照特定的顺序排列。

        # 通过卷积层获取qkv
        qkv = self.qkv_dwconv(self.qkv(x)) # 1C 变5C 然后 5C变5C
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # 将qkv分割成五个部分 5C变1C

        # 对v进行排序，并根据相同的索引更新q1, k1, q2, k2   B C (H W)
        v, idx = v.view(b, c, -1).sort(dim=-1)

        # torch.gather 函数用于根据索引 idx 从 q1 中收集特定位置的元素。    B C (H W)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)

        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        # 计算两次注意力机制
        out1 = self.reshape_attn(q1, k1, v, True)   # 局部信息
        out2 = self.reshape_attn(q2, k2, v, False)  # 长距离依赖关系

        # torch.scatter(input, dim, index, src)
        #       input: 目标张量，最终的结果会写入这个张量。
        #       dim: 沿着哪个维度进行散射（scatter）操作。
        #       index: 包含索引的张量，指定了 src 中的元素应该放置在 input 中的哪些位置。
        #       src: 源张量，从中取出元素并按照 index 指定的位置放置到 input 中。
        ##  B C (H W) --->B C H W
        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)

        out = out1 * out2  # 合并两个输出
        out = self.project_out(out)  # 投影输出

        out_replace = out[:, :c // 2]   # out_replace 的值复制到 out 张量的前半部分通道
        # 反向应用之前记录的排序索引 B C H W
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c // 2] = out_replace  # 更新输出

        return out  # 返回最终输出

if __name__ == "__main__":
    model = Dynamic_range_Histogram_SelfAttention(64)
    input = torch.randn(1, 64, 128, 128)

    output = model(input)

    print('Input size:', input.size())
    print('Output size:', output.size())
