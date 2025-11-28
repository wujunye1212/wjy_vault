import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

"""
    论文地址：https://arxiv.org/pdf/2503.06896
    论文题目：CATANet: Efﬁcient Content-Aware Token Aggregation for Lightweight Image Super-Resolution（CVPR 2025）
    中文题目：CATANet：用于轻量级图像超分辨率的高效内容感知令牌聚合网络 （CVPR 2025）
    讲解视频：https://www.bilibili.com/video/BV1Coo3YXEJB/
    令牌聚合块 (Token-Aggregation Block ,TAB)：
        实际意义：①传统聚类方法的局限性：聚类方法在实现细粒度长距离信息交互，但聚类质心特征更新会降低模型推理速度，利用聚类中心促进长距离信息传播，会引入无关信息。
                ②Transformer的不足：在处理图像超分辨率时，常将图像划分为小区域以降低计算复杂度，这种方式限制长距离相似令牌（特征）的使用。
        实现方式：①组内自注意力（IASA）在内容特征相似令牌（特征）之间实现精细信息交互。
                ②内容感知令牌聚合（CATA）模块，根据跨长距离特征相似性，形成更精确分组。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】   
"""
# 判断一个值是否不为 None
def exists(val):
    return val is not None

# 判断一个张量是否为空（元素数量为 0）
def is_empty(t):
    return t.nelement() == 0

# 在指定维度上扩展张量
def expand_dim(t, dim, k):
    # 在指定维度上增加一个维度
    t = t.unsqueeze(dim)
    # 初始化扩展形状，默认每个维度为 -1（表示保持原大小）
    expand_shape = [-1] * len(t.shape)
    # 将指定维度的大小设置为 k
    expand_shape[dim] = k
    # 扩展张量
    return t.expand(*expand_shape)

# 获取默认值，如果 x 不存在则使用默认值 d
def default(x, d):
    # 如果 x 不存在
    if not exists(x):
        # 如果 d 不是一个可调用对象，直接返回 d
        # 如果 d 是可调用对象，调用它并返回结果
        return d if not isinstance(d, Callable) else d()
    return x

# 指数移动平均（EMA）计算
def ema(old, new, decay):
    # 如果旧值不存在，直接返回新值
    if not exists(old):
        return new
    # 计算指数移动平均值
    return old * decay + new * (1 - decay)

# 原地进行指数移动平均更新
def ema_inplace(moving_avg, new, decay):
    # 如果移动平均值为空
    if is_empty(moving_avg):
        # 直接将新值复制给移动平均值
        moving_avg.data.copy_(new)
        return
    # 原地更新移动平均值
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

# 计算输入 x 和均值 means 之间的相似度
def similarity(x, means):
    # 使用爱因斯坦求和约定计算相似度
    return torch.einsum('bld,cd->blc', x, means)

# 计算距离和每个元素所属的桶
def dists_and_buckets(x, means):
    # 计算相似度作为距离
    dists = similarity(x, means)
    # 找到每个元素对应的最大相似度的索引，即所属的桶
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

# 批量计算每个类别在指定维度上的数量
def batched_bincount(index, num_classes, dim=-1):
    # 获取索引的形状
    shape = list(index.shape)
    # 将指定维度的大小设置为类别数量
    shape[dim] = num_classes
    # 初始化一个全零张量，用于存储每个类别的数量
    out = index.new_zeros(shape)
    # 在指定维度上，根据索引将 1 累加到对应的位置
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

# 迭代更新中心点
def center_iter(x, means, buckets=None):
    # 提取输入 x 的形状信息和数据类型，以及均值 means 的标记数量
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]

    # 如果没有提供 buckets
    if not exists(buckets):
        # 调用 dists_and_buckets 函数计算距离和每个元素所属的桶
        _, buckets = dists_and_buckets(x, means)

    # 计算每个桶中元素的数量，在批次维度上求和并保持维度
    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    # 创建一个掩码，标记哪些桶的元素数量为 0
    zero_mask = bins.long() == 0

    # 创建一个全零张量，用于存储每个桶的特征和
    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    # 根据 buckets 信息将 x 中的元素累加到对应的桶中
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    # 在批次维度上求和并进行归一化，最后转换数据类型
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    # 根据零掩码，选择使用原来的均值还是新计算的均值
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    # 移除批次维度
    means = means.squeeze(0)
    # 返回更新后的均值
    return means

class IASA(nn.Module):
    def __init__(self, dim, qk_dim, heads, group_size):
        # 调用父类的构造函数
        super().__init__()
        # 注意力头的数量
        self.heads = heads
        # 用于将输入映射到查询空间的线性层
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        # 用于将输入映射到键空间的线性层
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        # 用于将输入映射到值空间的线性层
        self.to_v = nn.Linear(dim, dim, bias=False)
        # 用于对输出进行投影的线性层
        self.proj = nn.Linear(dim, dim, bias=False)
        # 分组大小
        self.group_size = group_size

    def forward(self, normed_x, idx_last, k_global, v_global):
        # 获取归一化后的输入
        x = normed_x
        # 获取输入的批次大小、序列长度和特征维度
        B, N, _ = x.shape

        # 将输入映射到查询、键和值空间
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        # 根据 idx_last 对查询、键和值进行重排
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))

        # 确定分组大小，取序列长度和预设分组大小的最小值
        gs = min(N, self.group_size)
        # 计算分组数量
        ng = (N + gs - 1) // gs
        # 计算需要填充的元素数量
        pad_n = ng * gs - N

        # 对查询进行填充
        paded_q = torch.cat((q, torch.flip(q[:, N - pad_n:N, :], dims=[-2])), dim=-2)
        # 对填充后的查询进行形状重排
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d", ng=ng, h=self.heads)
        # 对键进行填充
        paded_k = torch.cat((k, torch.flip(k[:, N - pad_n - gs:N, :], dims=[-2])), dim=-2)
        # 使用 unfold 函数对填充后的键进行滑动窗口操作
        paded_k = paded_k.unfold(-2, 2 * gs, gs)
        # 对处理后的键进行形状重排
        paded_k = rearrange(paded_k, "b ng (h d) gs -> b ng h gs d", h=self.heads)
        # 对值进行填充
        paded_v = torch.cat((v, torch.flip(v[:, N - pad_n - gs:N, :], dims=[-2])), dim=-2)
        # 使用 unfold 函数对填充后的键进行滑动窗口操作
        paded_v = paded_v.unfold(-2, 2 * gs, gs)
        # 对处理后的键进行形状重排
        paded_v = rearrange(paded_v, "b ng (h d) gs -> b ng h gs d", h=self.heads)
        # 计算局部注意力输出
        out1 = F.scaled_dot_product_attention(paded_q, paded_k, paded_v)

        # 扩展全局键的维度以匹配批次和分组数量
        k_global = k_global.reshape(1, 1, *k_global.shape).expand(B, ng, -1, -1, -1)
        # 扩展全局值的维度以匹配批次和分组数量
        v_global = v_global.reshape(1, 1, *v_global.shape).expand(B, ng, -1, -1, -1)
        # 计算全局注意力输出
        out2 = F.scaled_dot_product_attention(paded_q, k_global, v_global)

        # 将局部和全局注意力输出相加
        out = out1 + out2
        # 对相加后的输出进行形状重排，并截取前 N 个元素
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]

        # 根据 idx_last 对输出进行重排
        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        # 通过投影层对输出进行处理
        out = self.proj(out)

        return out

class IRCA(nn.Module):
    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        # 注意力头的数量
        self.heads = heads
        # 用于将输入映射到键空间的线性层
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        # 用于将输入映射到值空间的线性层
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, normed_x, x_means):
        # 将归一化后的输入赋值给 x
        x = normed_x
        # 如果处于训练模式
        if self.training:
            # 调用 center_iter 函数，根据输入和均值计算全局特征
            x_global = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))
        else:
            # 如果处于推理模式，直接使用均值作为全局特征
            x_global = x_means

        # 通过线性层将全局特征映射到键和值空间
        k, v = self.to_k(x_global), self.to_v(x_global)
        # 使用 rearrange 函数将键的形状从 (n, h * dim_head) 重排为 (h, n, dim_head)
        k = rearrange(k, 'n (h dim_head)->h n dim_head', h=self.heads)
        # 使用 rearrange 函数将值的形状从 (n, h * dim_head) 重排为 (h, n, dim_head)
        v = rearrange(v, 'n (h dim_head)->h n dim_head', h=self.heads)

        # 返回重排后的键、值和全局特征（去除梯度信息）
        return k, v, x_global.detach()


class PreNorm(nn.Module):
    """Normalization layer.
    Args:
        dim (int): Base channels.
        fn (Module): Module after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

class Token_Aggregation_Block(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3,
                 num_tokens=8, group_size=128,
                 ema_decay=0.999):
        # 调用父类nn.Module的构造函数
        super().__init__()

        # 迭代次数
        self.n_iter = n_iter
        # 指数移动平均的衰减率
        self.ema_decay = ema_decay
        # 标记的数量
        self.num_tokens = num_tokens

        # 层归一化层，用于对输入进行归一化处理
        self.norm = nn.LayerNorm(dim)
        # 预归一化层，内部包含一个卷积前馈网络
        self.mlp = PreNorm(dim, ConvFFN(dim, mlp_dim))
        # IRCA注意力模块
        self.irca_attn = IRCA(dim, qk_dim, heads)
        # IASA注意力模块
        self.iasa_attn = IASA(dim, qk_dim, heads, group_size)
        # 注册一个可训练的缓冲区，用于存储标记的均值
        self.register_buffer('means', torch.randn(num_tokens, dim))
        # 注册一个可训练的缓冲区，用于标记是否已经初始化
        self.register_buffer('initted', torch.tensor(False))
        # 1x1卷积层，用于特征变换
        self.conv1x1 = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        # 获取输入张量的形状，分别为批次大小、通道数、高度和宽度
        _, _, h, w = x.shape
        # 将输入张量从(b, c, h, w)的形状重排为(b, h*w, c)的形状
        x = rearrange(x, 'b c h w->b (h w) c')
        # 保存输入的残差，用于后续的残差连接
        residual = x

        # 对输入进行层归一化处理
        x = self.norm(x)
        # 获取归一化后输入张量的形状，分别为批次大小、序列长度和通道数
        B, N, _ = x.shape

        # 创建一个从0到N-1的索引张量，并扩展到批次大小
        idx_last = torch.arange(N, device=x.device).reshape(1, N).expand(B, -1)
        # 如果还没有初始化
        if not self.initted:
            # 计算需要填充的数量，以保证序列长度能被num_tokens整除
            pad_n = self.num_tokens - N % self.num_tokens
            # 对输入进行填充，填充部分是输入的最后pad_n个元素的翻转
            paded_x = torch.cat((x, torch.flip(x[:, N - pad_n:N, :], dims=[-2])), dim=-2)
            # 计算填充后输入的均值
            x_means = torch.mean(rearrange(paded_x, 'b (cnt n) c->cnt (b n) c', cnt=self.num_tokens), dim=-2).detach()
        else:
            # 如果已经初始化，直接使用之前保存的均值
            x_means = self.means.detach()

        # 如果处于训练模式
        if self.training:
            with torch.no_grad():
                # 进行n_iter - 1次迭代更新均值
                for _ in range(self.n_iter - 1):
                    x_means = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))

        # 通过IRCA注意力模块计算全局键、全局值和更新后的均值
        k_global, v_global, x_means = self.irca_attn(x, x_means)

        with torch.no_grad():
            # 计算输入与均值之间的相似度得分
            x_scores = torch.einsum('b i c,j c->b i j',
                                    F.normalize(x, dim=-1),
                                    F.normalize(x_means, dim=-1))
            # 获取每个元素所属的类别索引
            x_belong_idx = torch.argmax(x_scores, dim=-1)

            # 对类别索引进行排序
            idx = torch.argsort(x_belong_idx, dim=-1)
            # 根据排序后的索引重新排列idx_last
            idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1)
        # 通过IASA注意力模块进行计算
        """
            组内自注意力（IASA）在内容特征相似令牌之间实现精细信息交互。
        """
        y = self.iasa_attn(x, idx_last, k_global, v_global)

        # 将输出从(b, h*w, c)的形状重排为(b, c, h, w)的形状
        y = rearrange(y, 'b (h w) c->b c h w', h=h).contiguous()
        # 通过1x1卷积层进行特征变换
        y = self.conv1x1(y)

        # 进行残差连接
        x = residual + rearrange(y, 'b c h w->b (h w) c')
        # 通过预归一化层和卷积前馈网络进行计算，并再次进行残差连接
        x = self.mlp(x, x_size=(h, w)) + x

        # 如果处于训练模式
        if self.training:
            with torch.no_grad():
                # 保存更新后的均值
                new_means = x_means
                # 如果还没有初始化
                if not self.initted:
                    # 更新均值缓冲区
                    self.means.data.copy_(new_means)
                    # 标记为已经初始化
                    self.initted.data.copy_(torch.tensor(True))
                else:
                    # 使用指数移动平均更新均值
                    ema_inplace(self.means, new_means, self.ema_decay)

        # 将输出从(b, h*w, c)的形状重排为(b, c, h, w)的形状并返回
        return rearrange(x, 'b (h w) c->b c h w', h=h)

if __name__ == '__main__':
    model = Token_Aggregation_Block(
        dim=64,
        qk_dim=64,
        mlp_dim=128,
        heads=4
    )
    input = torch.rand(1, 64, 50, 50)
    output = model(input)
    print(f"Input size:  {input.size()}")
    print(f"Output size: {output.size()}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")