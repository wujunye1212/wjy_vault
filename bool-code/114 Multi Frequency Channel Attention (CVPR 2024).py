import math
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    论文地址：https://openaccess.thecvf.com/content/CVPR2024/papers/Nam_Modality-agnostic_Domain_Generalizable_Medical_Image_Segmentation_by_Multi-Frequency_in_Multi-Scale_CVPR_2024_paper.pdf
    论文题目：Modality-agnostic Domain Generalizable Medical Image Segmentation by Multi-Frequency in Multi-Scale Attention(CVPR 2024)
    中文题目：基于多尺度多频率注意力机制的模态无关领域通用医学图像分割(CVPR 2024)
    讲解视频：https://www.bilibili.com/video/BV1VDAwetEFu/
        基于余弦变换的多频率通道注意力（Multi-Frequency Channel Attention，MFCA)：
            实际意义：研究发现医学图像在尺度和频率维度上的分布不同，多频率信息方差更大，两者相互独立且互补，这为模型设计提供了新方向。
                    部分研究利用2D-DCT等频率变换方法和注意力机制，增强局部和全局上下文信息，但现有医学图像分割方法常忽视两者结合优势。
            实现方式：利用 2D DCT 提取频率统计特征，生成通道注意力图，重新校准特征图。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
            相似思路参考：https://www.bilibili.com/video/BV1ohK5e6Eb4/
"""

def get_freq_indices(method):
    # 确保方法在指定的选项中
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    # 从方法名中提取频率数
    num_freq = int(method[3:])
    if 'top' in method:
        # 预定义的 top 频率索引
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        # 选择前 num_freq 个索引
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        # 预定义的 low 频率索引
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        # 选择前 num_freq 个索引
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        # 预定义的 bot 频率索引
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        # 选择前 num_freq 个索引
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        # 如果方法不在选项中，抛出异常
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h=7,
                 dct_w=7,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()

        # 确保频率分支数是有效的
        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        # 构造频率选择字符串
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        # 获取频率索引
        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        # 根据 DCT 大小调整索引
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # 确保 mapper_x 和 mapper_y 长度一致
        assert len(mapper_x) == len(mapper_y)

        # 初始化 DCT 权重
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))

        # 定义全连接层
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        # 定义自适应池化层
        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        # 初始化 DCT 滤波器
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        # 构建 DCT 滤波器
        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        # 计算 DCT 滤波器值
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def forward(self, x):
        # 获取输入的形状
        batch_size, C, H, W = x.shape

        x_pooled = x
        # 如果输入大小与 DCT 大小不匹配，进行自适应池化
        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        # 初始化频谱特征
        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            # 循环遍历模型的状态字典，该字典包含模型的所有参数。它寻找名称中包含 'dct_weight' 的参数。
            if 'dct_weight' in name:
                # 计算频谱特征：将输入与 DCT 权重参数逐元素相乘
                x_pooled_spectral = x_pooled * params
                # 累加池化频谱特征的平均值
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                # 累加池化频谱特征的最大值
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                # 累加池化频谱特征的最小值：通过取反后最大池化的方法实现
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)


        # 归一化频谱特征
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq

        # 通过全连接层生成注意力图
        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)

        # 计算最终的注意力图
        multi_spectral_attention_map = F.sigmoid(multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        # 将注意力图应用于输入
        return x * multi_spectral_attention_map.expand_as(x)

if __name__ == '__main__':
    # 假设输入张量的形状为 (batch_size, in_channels, height, width)
    batch_size = 8
    in_channels = 64
    height, width = 50, 50
    input_tensor = torch.randn(batch_size, in_channels, height, width)

    # ========自定义参数========
    # 定义 DCT 高度和宽度
    dct_h, dct_w = 7, 7  # 通常是 7x7 的 DCT 块
    frequency_branches = 16  # 频率分支数，必须是 [1, 2, 4, 8, 16, 32] 之一
    frequency_selection = 'top'  # 选择 'top' 频率，应与 get_freq_indices 函数支持的选项匹配
    reduction = 16  # reduction 参数用于控制通道数的缩减，通常设置为 16
    # ========自定义参数========
    # 初始化 MultiFrequencyChannelAttention 模块
    mfca = MultiFrequencyChannelAttention(
        in_channels=in_channels,
        dct_h=dct_h,
        dct_w=dct_w,
        frequency_branches=frequency_branches,
        frequency_selection=frequency_selection,
        reduction=reduction
    )
    output_tensor = mfca(input_tensor)
    # 输出形状
    print("输入张量形状:", input_tensor.shape)
    print("输出张量形状:", output_tensor.shape)