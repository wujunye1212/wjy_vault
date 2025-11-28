import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)  # 执行快速傅里叶变换,返回频域表示的复数张量(只包含正频率项),长度为T/2+1:  (B,T,C)-->(B,F,C)  F=T/2+1; 对于实数输入，正频率和负频率是共轭对称的，所以只需要存储一半的频率。
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1) # (1)abs(xf)取傅里叶变换结果xf的幅值,幅值表示了每个频率分量的强度; (2)对batchsize维度进行平均,得到跨多个样本的平均频率幅值,此时shape为[F,C];  (3)mean(-1):对通道c维度再取平均，结果形状变为[F],表示频率分量的平均强度
    frequency_list[0] = 0 # 将零频率（直流成分）的幅值设为0,零频率通常表示信号的直流分量,即信号的平均值,对周期性分析并不重要,因此将其排除
    _, top_list = torch.topk(frequency_list, k) # 使用torch.topk 函数从 frequency_list 中找到幅值最大的 k 个频率分量, 返回值是最大幅值对应的索引(频率位置), 存储到top_list
    top_list = top_list.detach().cpu().numpy() # 转换为numpy数组
    period = x.shape[1] // top_list # 将时间序列长度 T 除以对应频率位置 top_list,计算得到每个频率分量对应的周期; 频率与周期成反比关系.  (在离散傅里叶变换中,频率与周期成反比关系. 对于离散时间信号,频率索引与实际频率直接相关,索引越大,频率越高,周期越短. 因此，通过T除以索引i可以得到该频率分量的周期.)
    return period, abs(xf).mean(-1)[:, top_list] # period:周期;  abs(xf).mean(-1)[:, top_list]: shape为[B, k], 取出K个频率位置对应的K个幅值


class TimesBlock(nn.Module):
    def __init__(self, seq_len=96,pred_len=96,top_k=3,d_model=64,d_ff=64,num_kernels=3):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                               num_kernels=num_kernels)
        )
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.out_linear = nn.Linear(self.pred_len + self.seq_len, self.seq_len)

    def forward(self, x):
        # 将序列长度T映射到2T,包括历史和未来序列长度(self.pred_len + self.seq_len)
        x = self.predict_linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        B, T, N = x.size()

        period_list, period_weight = FFT_for_Period(x, self.k) # x:(B,T,N); period_list:[k], K个周期;  period_weight:[B,K],幅值

        res = []
        for i in range(self.k):
            period = period_list[i] # 取第i个周期
            # padding  如果T不能整除周期长度, 则填充0
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,N).permute(0, 3, 1, 2).contiguous()  # (B,T,N)--reshape-->(B,num,P,N)-permute-->(B,N,num,P)   将1维转换为2维, T = num * P, num是周期个数, P是周期长度
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out) # 执行inception Conv(具有多个不同卷积核大小的卷积层): (B,N,num,P)-->(B,N,num,P)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N) # 将out恢复与输出相同的shape: (B,N,num,P)--permute-->(B,num,P,N)-reshape-->(B,T,N)
            res.append(out[:, :(self.seq_len + self.pred_len), :]) # (B,T,N),选择前T个时间步作为输出,这是因为有的序列长度通过0填充,超过了T
        res = torch.stack(res, dim=-1) # 将K个输出进行拼接:(B,T,N,K)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1) # 在K维度上执行softmax,获得每一个频率对应的权重: (B,K)--softmax-->(B,K)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1) # 进行复制以产生和res相同的shape: (B,K)-unsqueeze--unsqueeze-->(B,1,1,K)--repeat-->(B,T,N,K)
        res = torch.sum(res * period_weight, -1) # 加权求和: (B,T,N,K) * (B,T,N,K) == (B,T,N,K);  (B,T,N,K)--sum-->(B,T,N)
        # residual connection
        res = res + x  # 添加残差连接:(B,T,N)
        res = self.out_linear(res.permute(0, 2, 1)).permute(0, 2, 1)  # 将(pred_len+seq_len)个时间步映射到pred_len个时间步
        return res


if __name__ == '__main__':
    #  (B,T,N)  B:batchsize; T:序列长度; N:通道数量
    x1 = torch.randn(1,96,64).to(device)
    B,T,N = x1.size()

    # seq_len:历史时间步长;  pred_len:预测时间步长;  top_k:选择K个频率;  d_model:通道;  d_ff:inception Conv中的通道;  num_kernels: inception中的卷积层个数
    Model = TimesBlock(seq_len=T,pred_len=T,top_k=5,d_model=N,d_ff=N,num_kernels=3).to(device)

    out = Model(x1) # (B,T,N)--> (B,T,N)
    print(out.shape)
