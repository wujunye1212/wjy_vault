import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange


'''
来自TGRS 2025顶刊
'''
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

class LMM(nn.Module):
    def __init__(self, channels):
        super(LMM, self).__init__()
        self.channels = channels
        dim = self.channels

        # 方向卷积：用于提取横向（3×7）和纵向（7×3）局部特征，采用深度可分离卷积（groups=dim）
        self.fc_h = nn.Conv2d(dim, dim, (3, 7), stride=1, padding=(1, 7 // 2), groups=dim, bias=False)
        self.fc_w = nn.Conv2d(dim, dim, (7, 3), stride=1, padding=(7 // 2, 1), groups=dim, bias=False)

        # 多层感知机：通道注意力模块，输入维度为dim，隐层维度为dim/2，输出为3倍通道数（用于三分支加权）
        self.reweight = Mlp(dim, dim // 2, dim * 3)

    def swish(self, x):
        # Swish激活函数：swish(x) = x * sigmoid(x)
        return x * torch.sigmoid(x)

    def forward(self, x):
        N, C, H, W = x.shape  # 获取输入维度
        # 提取方向特征：横向和纵向卷积
        x_w = self.fc_h(x)  # 横向信息
        x_h = self.fc_w(x)  # 纵向信息
        # 三路融合：将方向特征与原始输入相加（残差结构）
        x_add = x_h + x_w + x
        # 通道注意力：全局平均池化 -> MLP -> 重塑为3个通道权重
        att = F.adaptive_avg_pool2d(x_add, output_size=1)  # B, C, 1, 1
        att = self.reweight(att).reshape(N, C, 3).permute(2, 0, 1)  # 转换为(3, B, C)
        att = self.swish(att).unsqueeze(-1).unsqueeze(-1)  # 添加空间维度，变为(3, B, C, 1, 1)
        # 三路加权融合：方向分支和原始输入加权求和
        x_att = x_h * att[0] + x_w * att[1] + x * att[2]
        return x_att  # 输出融合后的局部增强特征
class GMM(nn.Module):
    def __init__(self, channels, H, W):
        super(GMM, self).__init__()
        self.channels = channels
        patch = 4  # 切片因子
        self.C = int(channels / patch)  # 每块切片的通道数

        # 行方向重排后输入通道为 H×C'，输出为 C'×H，使用3×3深度卷积
        self.proj_h = nn.Conv2d(H * self.C, self.C * H, (3, 3), stride=1, padding=(1, 1), groups=self.C, bias=True)
        # 列方向重排后输入通道为 W×C'，输出为 C'×W
        self.proj_w = nn.Conv2d(W * self.C, self.C * W, (3, 3), stride=1, padding=(1, 1), groups=self.C, bias=True)

        # 通道融合：用于将重排卷积结果与原始输入拼接后进行压缩
        self.fuse_h = nn.Conv2d(channels * 2, channels, (1, 1), (1, 1), bias=False)
        self.fuse_w = nn.Conv2d(channels * 2, channels, (1, 1), (1, 1), bias=False)

        # 相对位置编码（行/列方向）
        self.relate_pos_h = RelativePosition(channels, H)
        self.relate_pos_w = RelativePosition(channels, W)

        self.activation = nn.GELU()  # 激活函数
        self.BN = nn.BatchNorm2d(channels)  # 标准化

    def forward(self, x):
        N, C, H, W = x.shape
        # 生成行、列方向的位置编码
        pos_h = self.relate_pos_h(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        pos_w = self.relate_pos_w(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        C1 = int(C / self.C)  # 每个切片块的分组数

        # 行方向特征处理
        x_h = x + pos_h  # 加入位置编码
        x_h = x_h.view(N, C1, self.C, H, W)  # 划分为多个块
        x_h = x_h.permute(0, 1, 3, 2, 4).contiguous().view(N, C1, H, self.C * W)  # 重排为列方向特征
        x_h = self.proj_h(x_h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # 卷积处理
        x_h = x_h.view(N, C1, H, self.C, W).permute(0, 1, 3, 2, 4).contiguous().view(N, C, H, W)  # 还原形状
        x_h = self.fuse_h(torch.cat([x_h, x], dim=1))  # 与输入融合并压缩通道
        x_h = self.activation(self.BN(x_h)) + pos_w  # 激活 + 加入列方向位置编码

        # 列方向特征处理
        x_w = self.proj_w(x_h.view(N, C1, H * self.C, W).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_w = x_w.contiguous().view(N, C1, self.C, H, W).view(N, C, H, W)

        # 融合列方向输出与输入
        x = self.fuse_w(torch.cat([x, x_w], dim=1))
        return x  # 输出全局增强后的特征
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
'''二次创新模块:MGLFM 多尺度全局局部特征融合模块'''
class MGLFM(nn.Module):
    def __init__(self, dim, H,W):
        super(MGLFM, self).__init__()
        self.LLM = LMM(dim)
        self.GMM = GMM(dim, H,W)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        # LLM = self.LLM(initial)
        # GMM = self.GMM(initial)
        # pattn1 = LLM+GMM
        pattn1 = self.LLM(self.GMM(initial))
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input= torch.randn(1, 32, 64, 64)
    # 创建 GMM 模块
    gmm = GMM(channels=32,H=64,W=64)
    # 将输入图像传入GMM 模块进行处理
    output = gmm(input)
    # 输出结果的形状
    # 打印输入和输出的形状
    print('全局混合模块_GMM_input_size:', input.size())
    print('全局混合模块_GMM_output_size:', output.size())

    # 创建 LMM 模块
    lmm = LMM(channels=32)
    # 将输入图像传入LMM模块进行处理
    output = lmm(input)
    # 输出结果的形状
    # 打印输入和输出的形状
    print('局部混合模块LMM_input_size:', input.size())
    print('局部混合模块LMM_output_size:', output.size())

    #二次创新模块MGLFM 多尺度全局局部特征融合模块
    block = MGLFM(dim=32,H=64,W=64)
    input1 = torch.rand(1, 32, 64, 64)
    input2 = torch.rand(1, 32, 64, 64)
    output = block(input1, input2)
    print('二次创新模块—MGLFM_input_size:', input1.size())
    print('二次创新模块—MGLFM_output_size:', output.size())