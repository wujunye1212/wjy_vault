import torch
import torch.nn as nn


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

    def __init__(self, seq_len, pred_len, hidden_size):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = 0.02
        self.revin_layer = RevIN(hidden_size, affine=True, subtract_last=False)

        self.embed_size = self.seq_len
        self.hidden_size = hidden_size

        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

    def circular_convolution(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho') # (B,C,L)-->(B,C,D)
        w = torch.fft.rfft(w, dim=1, norm='ortho') # (1,L)-->(1,D)
        y = x * w  # 调节每个时间步信息: (B,C,D) * (1,D) == (B,C,D)
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho") # 逆傅里叶变换： (B,C,D)-->(B,C,L)
        return out

    def forward(self, x):
        z = x
        z = self.revin_layer(z, 'norm')
        x = z

        x = x.permute(0, 2, 1) # (B,L,C)-->(B,C,L)

        x = self.circular_convolution(x, self.w.to(x.device))  # x:(B,C,L); w:(1,L); 输出x: (B,C,L)

        x = self.fc(x) # (B,C,L)-->(B,C,L)
        x = x.permute(0, 2, 1) # (B,C,L)-->(B,L,C)

        z = x
        z = self.revin_layer(z, 'denorm')
        x = z

        return x


if __name__ == '__main__':

    # (B,L,C)
    x1 = torch.randn(1, 96, 64)
    B, L, C = x1.size()

    # 定义 PairFilter
    Model = Model(seq_len=L, pred_len=L, hidden_size=C)

    # 执行 PairFilter
    out = Model(x1)
    print(out.shape)