import torch
from torch import nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1,bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=1, groups=groups,bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=small_kernel,
                                            stride=stride, padding=small_kernel // 2, groups=groups, dilation=1,bias=False)


    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs) # (B,MD,N)-->(B,MD,N)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)

        return out



class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):

        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)

        #convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        #convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//dmodel
    def forward(self,x):

        input = x
        B, M, D, N = x.shape  # (B,M,D,N)

        # 跨时间; 大核DWConv,用于学习每个时间序列内部的时间依赖性.
        x = x.reshape(B,M*D,N) # 将通道D和变量M进行合并: (B,M,D,N)--reshape-->(B,MD,N)
        x = self.dw(x) # 执行DWConv,使特征和变量独立, 这样的话能够学习每个单变量时间序列的时间依赖性:(B,MD,N)--dw-->(B,MD,N)
        x = x.reshape(B,M,D,N) # (B,MD,N)-->(B,M,D,N)
        x = x.reshape(B*M,D,N) # (B,M,D,N)-->(BM,D,N)
        x = self.norm(x) # BatchNorm1d正则化
        x = x.reshape(B, M, D, N) # (BM,D,N)-->(B,M,D,N)
        x = x.reshape(B, M * D, N) # (B,M,D,N)-->(B,MD,N)

        # 跨特征(通道); 第一阶段ConvFFN,包含两个卷积层, 先升维后降维的瓶颈结构, 用于学习单个时间序列变量的新的特征.
        x = self.ffn1drop1(self.ffn1pw1(x))  # 执行DWConv,分M组, 因此学习的是每个变量的特征表示 (如果ffn_ratio大于1的话,那就是先升维): (B,MD,N)--ffn1pw1-->(B,Md,N);  d=ffn_ratio*D
        x = self.ffn1act(x) # GELU激活函数
        x = self.ffn1drop2(self.ffn1pw2(x)) # 执行DWConv,分M组, 因此学习的是每个变量的特征表示 (如果ffn_ratio大于1的话,那就是后降维): (B,Md,N)--ffn1pw2-->(B,MD,N)
        x = x.reshape(B, M, D, N) # (B,MD,N)-->(B,M,D,N)

        # 跨变量; 第二阶段ConvFFN,包含两个卷积层, 先升维后降维的瓶颈结构, 用于学习M个时间序列变量之间的依赖性.
        x = x.permute(0, 2, 1, 3)  # (B,M,D,N)-->(B,D,M,N)
        x = x.reshape(B, D * M, N) # (B,D,M,N)-->(B,DM,N)
        x = self.ffn2drop1(self.ffn2pw1(x)) # 执行DWConv,分D组,因此学习的是M个变量之间的依赖性,(先升维): (B,DM,N)--ffn2pw1-->(B,dM,N); d=ffn_ratio*D
        x = self.ffn2act(x) # GELU激活函数
        x = self.ffn2drop2(self.ffn2pw2(x)) # 执行DWConv,分D组,因此学习的是M个变量之间的依赖性,(后降维): (B,dM,N)--ffn2pw2-->(B,DM,N)
        x = x.reshape(B, D, M, N) # (B,DM,N)-->(B,D,M,N)
        x = x.permute(0, 2, 1, 3) # (B,D,M,N)-->(B,M,D,N)

        x = input + x # 添加残差连接
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars,
                 small_kernel_merged=False, drop=0.1):

        super(Stage, self).__init__()
        d_ffn = int(dmodel * ffn_ratio)
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x


if __name__ == '__main__':
    # (B, M, D, N)  B:batchsize, M:序列的个数  N:序列长度  D:通道的数量
    x1 = torch.randn(1,7,64,336).to(device)
    B,M,D,N = x1.size()

    # nvars==M; ffn_ratio:维度提高的倍数; num_blocks:深度; large_size:用于学习时间依赖性的卷积核; small_size:没用上
    Model = Stage(ffn_ratio=2, num_blocks=1, large_size=51, small_size=5, dmodel=D, dw_model=D, nvars=M).to(device)
    out= Model(x1) # out:  (T,B,ND)


    print(out.shape)