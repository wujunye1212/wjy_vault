import math
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

#看 Ai缝合怪 b站视频：2025.7.21 更新的视频
class FeedForward(nn.Module):
    """
    MLP block with pre-layernorm, GELU activation, and dropout.
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

#看 Ai缝合怪 b站视频：2025.7.21 更新的视频
class AttentionBlock(nn.Module):
    """
    Global multi-head self-attention block with optional projection.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = (#看 Ai缝合怪 b站视频：2025.7.21 更新的视频
            dim_head * heads
        )  # the total dimension used inside the multi-head attention. When concatenating all heads, the combined dimension is dim_head × heads
        project_out = not (
            heads == 1 and dim_head == dim
        )  # if we're using just 1 head and its dimension equals dim, then we can skip the final linear projection.

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)  # Applies LN over the last dimension.

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # 看 Ai缝合怪 b站视频：2025.7.21 更新的视频
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """
        Expected input shape: [B, L, C]
        """
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(
            3, dim=-1
        )  # chunk splits into 3 chuncks along the last dimension, this gives Q, K, V
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

#看 Ai缝合怪 b站视频：2025.7.21 更新的视频
class LocalAttention2D(nn.Module):
    """
    Windowed/local attention for 2D grids using unfold & fold.
    """

    def __init__(self, kernel_size, stride, dim, heads, dim_head, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride  # kernel_size
        self.dim = dim
        padding = 0

        self.norm = nn.LayerNorm(dim)

        self.Attention = AttentionBlock(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
        )

        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        x = rearrange(
            x, "B H W C -> B C H W"
        )  # Rearrange to [B, C, H, W] for unfolding

        # unfold into local 2D patches
        patches = self.unfold(x)  # [B, C*K*K, L] where W is the number of patches

        patches = rearrange(
            patches,
            "B (C K1 K2) L -> (B L) (K1 K2) C",
            K1=self.kernel_size,
            K2=self.kernel_size,
        )
        # 看 Ai缝合怪 b站视频：2025.7.21 更新的视频
        patches = self.norm(patches)

        # Intra-Window self.attention
        out = self.Attention(patches)  # [B*L, K*K, C]

        # Reshape back to [B, C*K*K, L]
        out = rearrange(
            out,
            "(B L) (K1 K2) C -> B (C K1 K2) L",
            B=B,
            K1=self.kernel_size,
            K2=self.kernel_size,
        )

        # Fold back to [B, C, H, W] with overlap
        fold = nn.Fold(
            output_size=(H, W), kernel_size=self.kernel_size, stride=self.stride
        )
        out = fold(out)

        # Normalize overlapping regions
        norm = self.unfold(torch.ones((B, 1, H, W), device=x.device))  # [B, K*K, L]
        norm = fold(norm)  # [B, 1, H, W]
        out = out / norm

        # Reshape to [B, H, W, C]
        out = rearrange(out, "B C H W -> B H W C")

        return out
#看 Ai缝合怪 b站视频：2025.7.21 更新的视频
class Multipole_Attention(nn.Module):  # B C H W 输入，定义多极注意力模块
    """
    多尺度层次化局部注意力机制（支持下采样和上采样）。
    """

    def __init__(
        self,
        in_channels,                        # 输入特征的通道数 C
        image_size,                         # 输入图像尺寸（假设为正方形）
        local_attention_kernel_size=2,      # 局部注意力窗口的大小
        local_attention_stride=2,           # 注意力窗口滑动步长
        downsampling="conv",                # 下采样方式：'avg_pool' 或 'conv'
        upsampling="conv",                  # 上采样方式：'avg_pool' 或 'conv'
        sampling_rate=2,                    # 下采样/上采样倍数
        heads=4,                            # 多头注意力的头数
        dim_head=16,                        # 每个注意力头的维度
        dropout=0.1,                        # dropout 概率
        channel_scale=1,                    # 多尺度通道数扩展系数
    ):
        super().__init__()
        # 看 Ai缝合怪 b站视频：2025.7.21 更新的视频

        # 固定使用5个金字塔层级（尺度），可考虑替换为：int(log(image_size, sampling_rate))
        self.levels = int(math.log(image_size, sampling_rate))

        # 定义每一层的通道数（默认都不变），如果 channel_scale > 1 可实现通道扩张
        channels_conv = [in_channels * (channel_scale**i) for i in range(self.levels)]

        # 定义局部注意力模块（这里只创建了一个 Attention 实例，共用于所有层）
        self.Attention = LocalAttention2D(
            kernel_size=local_attention_kernel_size,    # 注意力窗口大小
            stride=local_attention_stride,              # 滑动窗口步长
            dim=channels_conv[0],                       # 输入特征维度
            heads=heads,                                # 多头数
            dim_head=dim_head,                          # 每头的维度
            dropout=dropout                             # Dropout 概率
        )

        # ---------------- 下采样模块 ---------------- #
        if downsampling == "avg_pool":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),                             # 调整维度适应PyTorch卷积格式
                nn.AvgPool2d(kernel_size=sampling_rate, stride=sampling_rate), # 平均池化
                Rearrange("B C H W -> B H W C"),                             # 再转回 NHWC 格式
            )

        elif downsampling == "conv":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Conv2d(
                    in_channels=channels_conv[0],     # 输入通道
                    out_channels=channels_conv[0],    # 输出通道数不变
                    kernel_size=sampling_rate,        # 下采样卷积核大小
                    stride=sampling_rate,             # 步长等于倍数，控制缩放
                    bias=False
                ),
                Rearrange("B C H W -> B H W C"),
            )

        # ---------------- 上采样模块 ---------------- #
        if upsampling == "avg_pool":
            current = image_size
            # 检查图像尺寸能否被采样率整除
            for _ in range(self.levels):
                assert current % sampling_rate == 0, f"尺寸不可被整除"
                current = current // sampling_rate

            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Upsample(scale_factor=sampling_rate, mode="nearest"),  # 最近邻上采样
                Rearrange("B C H W -> B H W C"),
            )

        elif upsampling == "conv":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.ConvTranspose2d(                        # 反卷积（上采样）
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False
                ),
                Rearrange("B C H W -> B H W C"),
            )

    def forward(self, x):
        # 输入 x 形状为 [B, C, H, W]，先变为 NHWC 格式以适配后续操作
        x = x.permute(0, 2, 3, 1)  # -> [B, H, W, C]
        x_in = x                   # 保留原始输入以供逐层下采样使用

        x_out = []                 # 存储各层注意力输出
        x_out.append(self.Attention(x_in))  # 第0层注意力（原分辨率）

        # 多尺度下采样+注意力
        for l in range(1, self.levels):  # 从第1层到第L-1层
            x_in = self.down(x_in)                 # 下采样
            x_out_down = self.Attention(x_in)      # 在当前尺度应用注意力
            x_out.append(x_out_down)

        res = x_out.pop()  # 从最后一层开始向上融合

        # 融合多层特征，从最粗到最细逐层上采样并累加
        for l, out_down in enumerate(x_out[::-1]):
            res = out_down + (1 / (l + 1)) * self.up(res)

        return res.permute(0, 3, 2, 1)  # 返回为 [B, C, W, H]，建议改为 [B, C, H, W]

class Multipole_Attention_BHWC(nn.Module):  #B H W C Multipole Attention 多极注意力
    """
    Hierarchical local attention across multiple scales with down/up-sampling.
#看 Ai缝合怪 b站视频：2025.7.21 更新的视频
    """

    def __init__(
        self,
        in_channels,
        image_size,
        local_attention_kernel_size=2,
        local_attention_stride=2,
        downsampling="conv",
        upsampling= "conv",
        sampling_rate=2,
        heads=4,
        dim_head=16,
        dropout=0.1,
        channel_scale=1,
    ):
        super().__init__()

        self.levels = int(math.log(image_size, sampling_rate))  # math.log(x, base)
        # self.levels = 5
        channels_conv = [in_channels * (channel_scale**i) for i in range(self.levels)]

        # 看 Ai缝合怪 b站视频：2025.7.21 更新的视频
        # A shared local attention layer for all levels
        self.Attention = LocalAttention2D(
            kernel_size=local_attention_kernel_size,
            stride=local_attention_stride,
            dim=channels_conv[0],
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        if downsampling == "avg_pool":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.AvgPool2d(kernel_size=sampling_rate, stride=sampling_rate),
                Rearrange("B C H W -> B H W C"),
            )

        elif downsampling == "conv":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Conv2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

        if upsampling == "avg_pool":
            current = image_size

            for _ in range(self.levels):
                assert (
                    current % sampling_rate == 0
                ), f"Image size not divisible by sampling_rate size at level {_}: current={current}, sampling_ratel={sampling_rate}"
                # 看 Ai缝合怪 b站视频：2025.7.21 更新的视频
                current = current // sampling_rate

            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Upsample(scale_factor=sampling_rate, mode="nearest"),
                Rearrange("B C H W -> B H W C"),
            )

        elif upsampling == "conv":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.ConvTranspose2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

    def forward(self, x):
        # x: [B, H, W, C], returns the same shape
        # Level 0
        x_in = x

        x_out = []
        x_out.append(self.Attention(x_in))

        # Levels from 1 to L
        for l in range(1, self.levels):
            x_in = self.down(x_in)
            x_out_down = self.Attention(x_in)
            x_out.append(x_out_down)

        res = x_out.pop()
        for l, out_down in enumerate(x_out[::-1]):
            res = out_down + (1 / (l + 1)) * self.up(res)

        return res

class MultipoleBlock(nn.Module): # 多极注意力块
    """
    Transformer block stacking multiple Multipole_Attention2D + FeedForward layers.
    """

    def __init__(
        self,
        in_channels,
        image_size,
        kernel_size=2,  # Local attention patch size
        local_attention_stride=2,  # stride（与 kernel_size 相同）
        downsampling="conv",  # 使用卷积做下采样
        upsampling="conv",  # 使用反卷积做上采样
        sampling_rate=2,  # 每层下采样/上采样缩放因子
        depth=2,  # 堆叠层数
        heads=4,  # 注意力头数
        dim_head=16,  # 每个头的维度
        att_dropout=0.1,  # 注意力 dropout
        channel_scale=1,  # 多尺度通道扩展倍率（设为1保持通道数一致）

    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.layers = nn.ModuleList([])
        mlp_dim = int(4*in_channels)  # FeedForward中间层维度
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Multipole_Attention_BHWC(
                            in_channels,
                            image_size,
                            kernel_size,
                            local_attention_stride,
                            downsampling,
                            upsampling,
                            sampling_rate,
                            heads,
                            dim_head,
                            att_dropout,
                            channel_scale,
                        ),
                        FeedForward(in_channels, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        """
        Expected input shape: [B, H, W, C]
        """
        x = x.permute(0,2,3,1)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x).permute(0,3,1,2)

# ICCV 2025 Multipole_Attention模块的两个二次创新模块，
# MPSA多级稀疏注意力和MPMA多级曼巴注意力在我的二次创新模块交流群，可以直接去跑实验发小论文！
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input = torch.randn(1, 32, 64, 64)
    # 创建 Multipole_Attention模块
    MA = Multipole_Attention(in_channels=32,image_size=64)  #第一个模块 ：多极注意力
    # 将输入图像传入Multipole_Attention模块进行处理
    output = MA(input)
    # 输出结果的形状
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新_Multipole_Attention_input_size:', input.size())
    print('Ai缝合即插即用模块永久更新_Multipole_Attention_output_size:', output.size())

    MABlock = MultipoleBlock(in_channels=32,image_size=64)  #第二个模块 ：多极注意力块
    # 将输入图像传入MultipoleBlock模块进行处理
    output = MABlock (input)
    # 输出结果的形状
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新_MultipoleBlock_input_size:', input.size())
    print('Ai缝合即插即用模块永久更新_MultipoleBlock_output_size:', output.size())


