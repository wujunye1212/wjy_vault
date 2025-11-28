import torch
from torch import Tensor, nn
from torch.nn import functional as F
from abc import abstractmethod
'''
    论文地址：https://arxiv.org/pdf/2404.14757
    论文题目：SST: Multi-Scale Hybrid Mamba - Transformer Experts for Long - Short Range Time Series Forecasting
    中文题目：SST：用于长短期时间序列预测的多尺度混合Mamba - Transformer专家模型
    讲解视频：https://www.bilibili.com/video/BV1cVmBYvEQL/
        全局专家：Mamba基于输入选择性记忆相关模式并过滤无关噪声，有效处理长范围时间序列。（Mamba替换Transformer）
        【2D图像的改进】
'''
def silu(x):
    # 定义 SiLU 激活函数
    return x * F.sigmoid(x)

class RMSNorm(nn.Module):
    """
        Gated Root Mean Square Layer Normalization
        论文链接: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 设置 epsilon 值以防止除零
        self.weight = nn.Parameter(torch.ones(d))  # 初始化可学习参数

    def forward(self, x, z):
        # 前向传播函数
        x = x * silu(z)  # 应用 SiLU 激活函数
        # 计算归一化值并返回
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class Mamba2(nn.Module):
    def __init__(self, d_model: int,  # 模型维度 (D)
                 n_layer: int = 24,  # Mamba-2 层的数量
                 d_state: int = 128,  # 状态维度 (N)
                 d_conv: int = 4,  # 卷积核大小
                 expand: int = 2,  # 扩展因子 (E)
                 headdim: int = 64,  # 头维度 (P)
                 chunk_size: int = 64,  # 矩阵分块大小 (Q)
                 ):
        super().__init__()
        self.n_layer = n_layer  # 保存层数
        self.d_state = d_state  # 保存状态维度
        self.headdim = headdim  # 保存头维度
        self.chunk_size = chunk_size  # 保存分块大小

        self.d_inner = expand * d_model  # 计算扩展后的内部维度
        assert self.d_inner % self.headdim == 0, "self.d_inner 必须能被 self.headdim 整除"
        self.nheads = self.d_inner // self.headdim  # 计算头数

        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads  # 输入投影维度
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)  # 定义线性投影层

        conv_dim = self.d_inner + 2 * d_state  # 卷积维度
        # 定义一维卷积层
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1)
        self.dt_bias = nn.Parameter(torch.empty(self.nheads, ))  # 定义偏置参数
        self.A_log = nn.Parameter(torch.empty(self.nheads, ))  # 定义 A_log 参数
        self.D = nn.Parameter(torch.empty(self.nheads, ))  # 定义 D 参数
        self.norm = RMSNorm(self.d_inner)  # 定义 RMSNorm 层
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)  # 定义输出投影层

    def forward(self, u: Tensor):
        # 计算负指数的 A_log，用于参数化的状态空间模型
        A = -torch.exp(self.A_log)  # (nheads,)

        # 输入投影，将输入 u 映射到更高维度的特征空间
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)

        """
            局部特征提取
        """
        # 将投影结果分割为 z, xBC 和 dt
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        # 通过 softplus 激活函数处理 dt，并加上偏置
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # 通过一维卷积处理 xBC，捕获局部上下文信息
        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )  # (batch, seqlen, d_inner + 2 * d_state)

        # 将卷积结果分割为 x, B, C
        x, B, C = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1
        )

        """
            多头机制
            重塑 x 的形状以适应多头机制
        """
        _b, _l, _hp = x.shape
        _h = _hp // self.headdim
        _p = self.headdim
        x = x.reshape(_b, _l, _h, _p)

        """
            使用 ssd 函数进行复杂的序列状态计算
            函数用于处理复杂的序列状态更新，结合参数化的状态空间模型，捕获长序列中的依赖关系。
        """
        y = self.ssd(x * dt.unsqueeze(-1),
                     A * dt,
                     B.unsqueeze(2),
                     C.unsqueeze(2))

        # 将计算结果与输入 x 结合，应用可学习参数 D
        y = y + x * self.D.unsqueeze(-1)

        # 将 y 重塑回原始形状
        _b, _l, _h, _p = y.shape
        y = y.reshape(_b, _l, _h * _p)

        # 应用 RMSNorm 进行归一化，并使用 z 进行缩放
        y = self.norm(y, z)

        # 通过输出投影层将特征维度调整回输入维度
        y = self.out_proj(y)

        return y

    def segsum(self, x: Tensor) -> Tensor:
        # 计算分段和
        T = x.size(-1)
        device = x.device
        x = x[..., None].repeat(1, 1, 1, 1, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def ssd(self, x, A, B, C):
        # 计算复杂的序列状态
        chunk_size = self.chunk_size
        x = x.reshape(x.shape[0], x.shape[1] // chunk_size, chunk_size, x.shape[2], x.shape[3])
        B = B.reshape(B.shape[0], B.shape[1] // chunk_size, chunk_size, B.shape[2], B.shape[3])
        C = C.reshape(C.shape[0], C.shape[1] // chunk_size, chunk_size, C.shape[2], C.shape[3])
        A = A.reshape(A.shape[0], A.shape[1] // chunk_size, chunk_size, A.shape[2])
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. 计算每个块内的输出 (对角块)
        L = torch.exp(self.segsum(A))
        Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

        # 2. 计算每个块内的状态
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

        # 3. 计算块间的 SSM 递归
        initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))[0]
        new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
        states = new_states[:, :-1]

        # 4. 计算状态到输出的转换
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

        # 合并块内和块间的输出
        Y = Y_diag + Y_off
        Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], Y.shape[3], Y.shape[4])

        return Y

class _BiMamba2(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 d_model: int,  # 模型维度 (D)
                 n_layer: int = 24,  # Mamba-2 层的数量
                 d_state: int = 128,  # 状态维度 (N)
                 d_conv: int = 4,  # 卷积核大小
                 expand: int = 2,  # 扩展因子 (E)
                 headdim: int = 64,  # 头维度 (P)
                 chunk_size: int = 64,  # 矩阵分块大小 (Q)
                 ):
        super().__init__()
        self.fc_in = nn.Linear(cin, d_model, bias=False)  # 调整通道数到 d_model
        self.mamba2_for = Mamba2(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size)  # 正向 Mamba2
        self.mamba2_back = Mamba2(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size)  # 反向 Mamba2
        self.fc_out = nn.Linear(d_model, cout, bias=False)  # 调整通道数到 cout
        self.chunk_size = chunk_size  # 保存分块大小

    @abstractmethod
    def forward(self, x):
        # 定义抽象的前向传播方法
        pass

class BiMamba2_2D(_BiMamba2):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__(cin, cout, d_model, **mamba2_args)
        self.fc_in = torch.nn.Linear(cin, d_model)
        self.fc_out = torch.nn.Linear(d_model, cout)

    def forward(self, x):
        h, w = x.shape[2:]
        x = F.pad(x, (0, (8 - x.shape[3] % 8) % 8,
                      0, (8 - x.shape[2] % 8) % 8))  # 将 h , w  pad到8的倍数, [b, c64, h8, w8]
        _b, _c, _h, _w = x.shape

        x = x.permute(0, 2, 3, 1).reshape(_b, _h * _w, _c)

        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1 = self.mamba2_for(x)
        x2 = self.mamba2_back(x.flip(1)).flip(1)

        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数

        x = x.reshape(_b, _h, _w, -1).permute(0, 3, 1, 2)  # 恢复到 (batch, channel, height, width)
        x = x[:, :, :h, :w]  # 截取原图大小
        return x

if __name__ == '__main__':
    net2 = BiMamba2_2D(64, 128, 64)
    x2 = torch.randn(1, 64, 32, 32)
    y2 = net2(x2)
    print(y2.shape)

    print("抖音、B站、小红书、CSDN同号")  # 打印社交媒体账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 打印提醒信息