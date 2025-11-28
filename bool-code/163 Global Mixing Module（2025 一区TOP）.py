import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

"""
    论文地址：https://ieeexplore.ieee.org/abstract/document/10817647/
    论文题目：STMNet: Single-Temporal Mask-based Network for Self-Supervised Hyperspectral Change Detection （2025 一区TOP）
    中文题目：STMNet：基于单时相掩膜的自监督高光谱变化检测网络（2025 一区TOP）
    讲解视频：https://www.bilibili.com/video/BV1jTMszPEFU/
        全局混合模块（Global Mixing Module，GMM）：
            实际意义：①长距离依赖不足：传统卷积难捕捉图像中远距离像素间关系。
                    ②全局信息利用弱：难以融合空间全局上下文与光谱特征的跨区域一致性。
                    ③特征方向受限：无法区分行列方向信息，对方向性特征不敏感。
            实现方式：①行列重组：按通道分组后，拉近远距离像素特征。
                    ②跨区卷积：3×3卷积提取重组后特征的跨区域信息，突破局部限制。
                    ③特征融合：通过全连接和激活函数，融合卷积特征与原始输入，强化全局关联
"""


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)
        trunc_normal_(self.embeddings_table, std=.02)

    def forward(self, length_q, length_k):
        # H, W
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]
        return embeddings

class GMM(nn.Module):
    """GMM (Global Mixed Model) 模块实现，用于特征融合和增强"""
    def __init__(self, channels, H, W):
        super(GMM, self).__init__()
        self.channels = channels
        # patch表示将通道分割的块数
        patch = 4
        # 计算每个分割块的通道数
        self.C = int(channels / patch)

        # 水平方向的投影卷积，保持空间尺寸不变
        self.proj_h = nn.Conv2d(H * self.C, self.C * H, (3, 3), stride=1, padding=(1, 1), groups=self.C, bias=True)
        # 垂直方向的投影卷积，保持空间尺寸不变
        self.proj_w = nn.Conv2d(W * self.C, self.C * W, (3, 3), stride=1, padding=(1, 1), groups=self.C, bias=True)

        # 水平方向特征融合的1x1卷积
        self.fuse_h = nn.Conv2d(channels * 2, channels, (1, 1), (1, 1), bias=False)
        # 垂直方向特征融合的1x1卷积
        self.fuse_w = nn.Conv2d(channels * 2, channels, (1, 1), (1, 1), bias=False)

        # 水平方向的相对位置编码
        self.relate_pos_h = RelativePosition(channels, H)
        # 垂直方向的相对位置编码
        self.relate_pos_w = RelativePosition(channels, W)

        # 激活函数和批归一化层
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(channels)

    def forward(self, x):
        # 获取输入张量的形状
        N, C, H, W = x.shape
        # 获取水平和垂直方向的相对位置编码，并调整维度
        pos_h = self.relate_pos_h(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        pos_w = self.relate_pos_w(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        # 计算通道分割的倍数
        C1 = int(C / self.C)

        # 水平方向处理
        # 添加水平方向的相对位置编码
        x_h = x + pos_h
        # 重塑张量，准备进行通道分组处理
        x_h = x_h.view(N, C1, self.C, H, W)
        # 调整维度顺序，进行水平方向的处理
        x_h = x_h.permute(0, 1, 3, 2, 4).contiguous().view(N, C1, H, self.C * W)
        # 应用水平方向的投影卷积
        x_h = self.proj_h(x_h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # 恢复原始维度顺序
        x_h = x_h.view(N, C1, H, self.C, W).permute(0, 1, 3, 2, 4).contiguous().view(N, C, H, W)
        # 融合处理后的特征和原始特征
        x_h = self.fuse_h(torch.cat([x_h, x], dim=1))

        # 应用批归一化和激活函数，并添加垂直方向的相对位置编码
        x_h = self.activation(self.BN(x_h)) + pos_w
        # 垂直方向处理
        # 重塑张量，准备进行垂直方向的处理
        x_w = self.proj_w(x_h.view(N, C1, H * self.C, W).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        # 恢复原始维度
        x_w = x_w.contiguous().view(N, C1, self.C, H, W).view(N, C, H, W)

        # 融合原始特征和垂直处理后的特征
        x = self.fuse_w(torch.cat([x, x_w], dim=1))
        return x

if __name__ == '__main__':
    input= torch.randn(1, 32, 50, 50)
    model = GMM(channels=32,H=50,W=50)
    output = model(input)
    print(f"输入张量形状: {input.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")