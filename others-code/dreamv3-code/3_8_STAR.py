import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape  # (B,D,L)

        # set FFN
        combined_mean = F.gelu(self.gen1(input)) # (B,D,L)-->(B,D,L)
        combined_mean = self.gen2(combined_mean) # (B,D,L)-->(B,D,L_core)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1) # 在通道方向上执行softmax,为随机池化生成一个概率权重: (B,D,L_core)-->(B,D,L_core)
            ratio = ratio.permute(0, 2, 1) # (B,D,L_core)--permute->(B,L_core,D)
            ratio = ratio.reshape(-1, channels) # 转换为2维, 便于进行采样: (B,L_core,D)--reshape-->(B*L_core,D)
            indices = torch.multinomial(ratio, 1) # 从多项分布ratio的每一行中抽取一个样本,返回值是采样得到的类别的索引: (B*L_core,1); 输入如果是一维张量,它表示每个类别的概率;如果是二维张量,每行表示一个概率分布
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1) # (B*L_core,1)--view--> (B,L_core,1)--permute-->(B,1,L_core)
            combined_mean = torch.gather(combined_mean, 1, indices) # 根据索引indices在D方向上选择对应的通道元素(理解为:选择重要的通道信息): (B,D,L_core)--gather-->(B,1,L_core)    # gather函数不了解的看这个:https://zhuanlan.zhihu.com/p/661293803
            combined_mean = combined_mean.repeat(1, channels, 1) # 复制D份,将随机选择的core表示应用到所有通道上: (B,1,L_core)--repeat-->(B,D,L_core)
        else:
            weight = F.softmax(combined_mean, dim=1) # 处于非训练模式时, 首先通过softmax生成一个权重分布:(B,D,L_core)-->(B,D,L_core)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1) # 直接在D方向上进行加权求和, 然后复制D份: (B,D,L_core)--sum-->(B,1,L_core)--repeat-->(B,D,L_core)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1) # (B,D,L)--cat--(B,D,L_core)==(B,D,L+L_core)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat)) # (B,D,L+L_core)-->(B,D,L)
        combined_mean_cat = self.gen4(combined_mean_cat) # (B,D,L)-->(B,D,L)
        output = combined_mean_cat

        return output



if __name__ == '__main__':
    #  batch_size, channels, d_series  (B,D,L)
    x1 = torch.randn(1,64,96).to(device)
    B,D,L = x1.size()

    Model = STAR(d_series=L, d_core=D).to(device)

    out = Model(x1) # (B,D,L)-->(B,D,L)
    print(out.shape)