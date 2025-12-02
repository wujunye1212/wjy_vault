import torch
import torch.nn as nn
import numpy as np

"""
    论文地址：https://arxiv.org/abs/2412.09319
    论文题目：FAMNet: Frequency-aware Matching Network for Cross-domain Few-shot Medical Image Segmentation（AAAI2025）
    中文题目：FAMNet：用于跨域少样本医学图像分割的频率感知匹配网络（AAAI2025）
    讲解视频：https://www.bilibili.com/video/BV12zQbY8Ekx/
        二维快速傅里叶变换（Two-dimensional Fast Fourier Transform，2D FFT）：
            实际意义：在图像分割任务里，不同物体（组织和病变）在图像中的表现形式复杂多样，对应不同频率特征。
            实现方式（优势）：①低频成分通常反映图像的整体轮廓信息，如器官的大致形状和位置；
                           ②中频成分包含更多结构和纹理信息，可用于区分不同组织的边界和特征；
                           ③高频成分则与图像的细节、边缘以及噪声相关。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class Two_dimensional_Fast_Fourier_Transform(nn.Module):
    def __init__(self):
        super(Two_dimensional_Fast_Fourier_Transform, self).__init__()

    def forward(self, x):
        # 调用filter_frequency_bands方法对输入x进行频率带滤波，得到低、中、高频特征
        fts_low, fts_mid, fts_high = self.filter_frequency_bands(x, cutoff=0.30)
        # 返回低、中、高频特征
        return fts_low, fts_mid , fts_high

    # 将张量重塑为方形的方法
    def reshape_to_square(self, tensor):
        # 获取输入张量的批量大小B、通道数C和元素数量N
        B, C, N = tensor.shape
        # 计算能容纳N个元素的最小方形边长
        side_length = int(np.ceil(np.sqrt(N)))
        # 计算方形张量的元素总数
        padded_length = side_length ** 2
        # 创建一个全零的填充张量，形状为(B, C, padded_length)，并放在与输入张量相同的设备上
        padded_tensor = torch.zeros((B, C, padded_length), device=tensor.device)
        # 将输入张量复制到填充张量的前N个位置
        padded_tensor[:, :, :N] = tensor
        # 将填充张量重塑为方形张量，形状为(B, C, side_length, side_length)
        square_tensor = padded_tensor.view(B, C, side_length, side_length)
        # 返回方形张量、边长、边长和原始元素数量
        return square_tensor, side_length, side_length, N

    # 对输入张量进行频率带滤波的方法
    def filter_frequency_bands(self, tensor, cutoff=0.2):
        device = tensor.device
        # 将输入张量的数据类型转换为float
        tensor = tensor.float()
        # 调用reshape_to_square方法将输入张量转换为方形张量，并获取相关信息
        tensor, H, W, N = self.reshape_to_square(tensor)

        # 获取方形张量的批量大小B和通道数C
        B, C, _, _ = tensor.shape

        # 计算方形张量中心到角点的最大半径
        max_radius = np.sqrt((H // 2) ** 2 + (W // 2) ** 2)
        # 计算低频截止半径
        low_cutoff = max_radius * cutoff
        # 计算高频截止半径
        high_cutoff = max_radius * (1 - cutoff)
        # 对输入张量进行二维快速傅里叶变换，并将低频分量移到频谱中心
        fft_tensor = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)), dim=(-2, -1))

        # 定义一个内部函数，用于创建频率滤波器
        def create_filter(shape, low_cutoff, high_cutoff, mode='band', device=device):
            # 获取滤波器的行数和列数
            rows, cols = shape
            # 计算滤波器的中心点坐标
            center_row, center_col = rows // 2, cols // 2
            # 在目标设备上创建网格坐标
            y, x = torch.meshgrid(
                torch.arange(rows, device=device),
                torch.arange(cols, device=device),
                indexing='ij'
            )
            # 计算每个点到中心点的欧氏距离
            distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)
            # 创建一个全零的掩码张量
            mask = torch.zeros((rows, cols), dtype=torch.float32, device=device)
            # 根据不同的模式设置掩码
            if mode == 'low':
                # 低频滤波器，保留中心圆形区域
                mask[distance <= low_cutoff] = 1
            elif mode == 'high':
                # 高频滤波器，保留外围环形区域
                mask[distance >= high_cutoff] = 1
            elif mode == 'band':
                # 中频带通滤波器，保留中间环形区域
                mask[(distance > low_cutoff) & (distance < high_cutoff)] = 1
            # 返回掩码
            return mask

        # 创建低频滤波器，并增加维度以匹配张量形状
        low_pass_filter = create_filter((H, W), low_cutoff, None, mode='low')[None, None, :, :]
        # 创建高频滤波器，并增加维度以匹配张量形状
        high_pass_filter = create_filter((H, W), None, high_cutoff, mode='high')[None, None, :, :]
        # 创建中频滤波器，并增加维度以匹配张量形状
        mid_pass_filter = create_filter((H, W), low_cutoff, high_cutoff, mode='band')[None, None, :, :]

        # 对傅里叶变换后的张量进行低频滤波
        low_freq_fft = fft_tensor * low_pass_filter
        # 对傅里叶变换后的张量进行高频滤波
        high_freq_fft = fft_tensor * high_pass_filter
        # 对傅里叶变换后的张量进行中频滤波
        mid_freq_fft = fft_tensor * mid_pass_filter

        # 对低频滤波后的张量进行逆傅里叶变换，并取实部
        low_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(low_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        # 对高频滤波后的张量进行逆傅里叶变换，并取实部
        high_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        # 对中频滤波后的张量进行逆傅里叶变换，并取实部
        mid_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(mid_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real

        """
            当 H*W == N 时：可以去掉，此时切片无实际作用（例如输入原本就是方形且不需要填充）

            当 H*W > N 时：
                必须保留：否则会保留无效的填充数据（可能包含零值或随机值），导致后续计算错误
                维度不匹配：可能导致输出张量形状与预期不符（例如期望 [B, C, 100]，实际输出 [B, C, 144]）
        """
        # 将方形的低频特征张量恢复为原始形状
        low_freq_tensor = low_freq_tensor.view(B, C, H * W)[:, :, :N]
        # 将方形的高频特征张量恢复为原始形状
        high_freq_tensor = high_freq_tensor.view(B, C, H * W)[:, :, :N]
        # 将方形的中频特征张量恢复为原始形状
        mid_freq_tensor = mid_freq_tensor.view(B, C, H * W)[:, :, :N]

        # 返回低、中、高频特征张量
        return low_freq_tensor, mid_freq_tensor, high_freq_tensor

if __name__ == '__main__':
    # 定义批量大小
    batch_size = 8
    # 定义特征维度 H  W
    feature_dim = 784
    # 定义元素数量
    channel = 64
    # 生成随机输入张量
    input = torch.rand(batch_size, feature_dim, channel)
    print("input1.shape:", input.shape)
    block = Two_dimensional_Fast_Fourier_Transform()

    low_freq_tensor, mid_freq_tensor, high_freq_tensor = block(input)
    print("low_freq_tensor.shape:", low_freq_tensor.shape)
    print("mid_freq_tensor.shape:", mid_freq_tensor.shape)
    print("high_freq_tensor.shape:", high_freq_tensor.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")