import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Model(nn.Module):

    def __init__(self, seq_len, pred_len, hidden_size, embed_size, dropout):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.band_width = 96
        self.scale = 0.02
        self.sparsity_threshold = 0.01

        self.revin_layer = RevIN(hidden_size, affine=True, subtract_last=False)
        self.embedding = nn.Linear(self.seq_len, self.embed_size)
        self.token = nn.Conv1d(in_channels=self.seq_len, out_channels=self.embed_size, kernel_size=(1,))

        self.w = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))

        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size)
        )

        self.output = nn.Linear(self.embed_size, self.pred_len)
        self.layernorm = nn.LayerNorm(self.embed_size)
        self.layernorm1 = nn.LayerNorm(self.embed_size)
        self.dropout = nn.Dropout(self.dropout)

    def tokenEmbed(self, x):
        x = self.token(x)
        return x

    # 数据依赖的频域滤波器生成器
    def texfilter(self, x):
        B, N, _ = x.shape #

        o1_real = F.relu(
            torch.einsum('bid,d->bid', x.real, self.w[0]) - \
            torch.einsum('bid,d->bid', x.imag, self.w[1]) + \
            self.rb1
        ) # 做复数乘法，并添加偏置

        o1_imag = F.relu(
            torch.einsum('bid,d->bid', x.imag, self.w[0]) + \
            torch.einsum('bid,d->bid', x.real, self.w[1]) + \
            self.ib1
        )

        o2_real = (
                torch.einsum('bid,d->bid', o1_real, self.w1[0]) - \
                torch.einsum('bid,d->bid', o1_imag, self.w1[1]) + \
                self.rb2
        )

        o2_imag = (
                torch.einsum('bid,d->bid', o1_imag, self.w1[0]) + \
                torch.einsum('bid,d->bid', o1_real, self.w1[1]) + \
                self.ib2
        )

        y = torch.stack([o2_real, o2_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, L, C = x.shape
        z = x
        z = self.revin_layer(z, 'norm')
        x = z

        x = x.permute(0, 2, 1) # (B,L,C)-->(B,C,L)
        x = self.embedding(x)  # 时间长度 L被映射到 D: (B,C,L)-->(B,C,D)
        x = self.layernorm(x) # 执行 layernorm
        x = torch.fft.rfft(x, dim=1, norm='ortho') # 执行傅里叶变换, c=(C+1)/2 :(B,C,D)-->(B,c,D)

        weight = self.texfilter(x) # 生成上下文滤波器
        x = x * weight # 上下文滤波器调整特征x
        x = torch.fft.irfft(x, n=C, dim=1, norm="ortho") # 逆傅里叶变换回时域表示
        x = self.layernorm1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.output(x)
        x = x.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, 'denorm')
        x = z

        return x


if __name__ == '__main__':

    # (B,L,C)
    x1 = torch.randn(1, 96, 64)
    B, L, C = x1.size()
    D = 128 #

    # 定义 TextFilter
    Model = Model(seq_len=L, pred_len=L, hidden_size=C, embed_size=D, dropout=0.2)

    # 执行 TextFilter
    out = Model(x1)
    print(out.shape)