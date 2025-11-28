import torch
import torch.nn as nn
from typing import Tuple
"""
    论文地址：https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_RMT_Retentive_Networks_Meet_Vision_Transformers_CVPR_2024_paper.pdf
    论文题目：RMT: Retentive Networks Meet Vision Transformers  (CVPR 2024)
    中文题目：RMT：保留网络与视觉变换器的结合 (CVPR 2024)
    讲解视频：https://www.bilibili.com/video/BV1yJ6gY3EHG/
        曼哈顿注意力
            扩展了RetNet的时间衰减机制到空间域，并基于曼哈顿距离提出了一个空间衰减矩阵，以引入自注意力中的明确空间先验。
            此外，还提出了一个适应于明确空间先验的注意力分解形式，旨在降低建模全局信息的计算负担，同时不影响空间衰减矩阵。
"""
class RetNetRelPos2d(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))  # 计算用于位置编码的角度
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))  # 计算衰减因子
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        index_h = torch.arange(H).to(self.decay)  # 生成高度索引
        index_w = torch.arange(W).to(self.decay)  # 生成宽度索引
        grid = torch.meshgrid([index_h, index_w])  # 创建索引网格
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # 将网格转换为 (H*W, 2) 的形状
        mask = grid[:, None, :] - grid[None, :, :]  # 计算曼哈顿距离
        mask = (mask.abs()).sum(dim=-1)  # 对最后一个维度求和
        mask = mask * self.decay[:, None, None]  # 将距离与衰减因子相乘
        return mask

    def generate_1d_decay(self, l: int):
        index = torch.arange(l).to(self.decay)  # 生成一维索引
        mask = index[:, None] - index[None, :]  # 计算一维曼哈顿距离
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]  # 将距离与衰减因子相乘
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1))  # 计算正弦位置编码
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1))  # 计算余弦位置编码
            retention_rel_pos = ((sin, cos), self.decay.exp())  # 结合衰减因子

        elif chunkwise_recurrent:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)  # 生成像素位置索引
            sin = torch.sin(index[:, None] * self.angle[None, :])  # 计算正弦位置编码
            sin = sin.reshape(slen[0], slen[1], -1)  # 重塑为三维张量

            cos = torch.cos(index[:, None] * self.angle[None, :])  # 计算余弦位置编码
            cos = cos.reshape(slen[0], slen[1], -1)  # 重塑为三维张量

            mask_h = self.generate_1d_decay(slen[0])  # 生成H方向的衰减掩码
            mask_w = self.generate_1d_decay(slen[1])  # 生成W方向的衰减掩码

            retention_rel_pos = ((sin, cos), (mask_h, mask_w))  # 组合位置编码与衰减掩码

        else:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(slen[0], slen[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(slen[0], slen[1], -1)
            mask = self.generate_2d_decay(slen[0], slen[1])  # 生成二维衰减掩码
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]  # 取出偶数维度
    x2 = x[:, :, :, :, 1::2]  # 取出奇数维度
    x = torch.stack([-x2, x1], dim=-1)  # 旋转操作
    out = x.flatten(-2)  # 展平
    return out

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)  # 调制特征的相对位置

class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)  # 转置为 (b c h w)
        x = self.conv(x)  # 执行卷积操作
        x = x.permute(0, 2, 3, 1)  # 转置回 (b h w c)
        return x

class VisionRetentionChunk(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos):
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos  # 提取位置编码、掩码

        q = self.q_proj(x)  # 计算查询向量
        k = self.k_proj(x)  # 计算键向量
        v = self.v_proj(x)  # 计算值向量

        lepe = self.lepe(v)  # 执行局部增强卷积 最后一步进行叠加

        k *= self.scaling  # 缩放键向量
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # 调整查询向量形状
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # 调整键向量形状
        qr = theta_shift(q, sin, cos)  # 应用位置调制
        kr = theta_shift(k, sin, cos)  # 应用位置调制

        qr_w = qr.transpose(1, 2)  # 转置以便在宽度方向上计算
        kr_w = kr.transpose(1, 2)  # 转置以便在宽度方向上计算
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)  # 调整值向量形状
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)  # 计算宽度方向的注意力
        qk_mat_w = qk_mat_w + mask_w  # 加上宽度方向的掩码
        qk_mat_w = torch.softmax(qk_mat_w, -1)  # 归一化
        v = torch.matmul(qk_mat_w, v)  # 应用注意力权重

        qr_h = qr.permute(0, 3, 1, 2, 4)  # 转置以便在高度方向上计算
        kr_h = kr.permute(0, 3, 1, 2, 4)  # 转置以便在高度方向上计算
        v = v.permute(0, 3, 2, 1, 4)  # 调整值向量形状
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)  # 计算高度方向的注意力
        qk_mat_h = qk_mat_h + mask_h  # 加上高度方向的掩码
        qk_mat_h = torch.softmax(qk_mat_h, -1)  # 归一化
        output = torch.matmul(qk_mat_h, v)  # 应用注意力权重

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # 调整输出形状

        output = output + lepe  # 加上局部增强的输出
        output = self.out_proj(output)  # 线性投影输出
        return output

if __name__ == '__main__':
    # 注意这里的形状为 B*H*W*C，而不是B*C*H*W
    inputs = torch.randn(1,32,32,64)  # 输入张量
    b, h, w, c = inputs.size()

    pos = RetNetRelPos2d(embed_dim=64, num_heads=4, initial_value=1, heads_range=3)  # 初始化位置编码
    rel_pos = pos((h, w), chunkwise_recurrent=True)  # 计算相对位置编码
    print(rel_pos)

    Model = VisionRetentionChunk(embed_dim=64, num_heads=4)  # 初始化模型
    out = Model(inputs, rel_pos)  # 执行前向传播
    print(out.shape)  # 输出结果形状
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")