import torch
from torch import nn

"Efficient Multi-Scale Attention Module with Cross-Spatial Learning"

class EMA(nn.Module):
    def __init__(self, channels, N=10, c2=None, factor=16):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)

        self.pool_c = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_t = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Conv2d(N, N, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.agp = nn.AdaptiveAvgPool2d((1, None))


    def forward(self, x):
        #(B,N,T,C)
        #b, c, h, w = x.size()
        B,N,T,C = x.size()
        channel_group = C // self.groups # 计算每一组的通道

        ### 坐标注意力模块  ###
        group_x = x.reshape(B * self.groups, N, T, -1)  # 在通道方向上将输入分为G组: (B,N,T,C)-->(B*G,N,T,C/G)
        x_t = self.pool_t(group_x)  # 使用全局平均池化压缩时间方向: (B*G,N,T,C/G)-->(B*G,N,1,C/G)
        x_c = self.pool_c(group_x).permute(0, 1, 3,2)  # 使用全局平均池化压缩通道方向: (B*G,N,T,C/G)-->(B*G,N,T,1)--permute->(B*G,N,1,T)
        tc = self.conv1x1(torch.cat([x_t, x_c], dim=-1))  # 将时间方向和通道方向的全局特征进行拼接: (B*G,N,1,C/G+T), 然后通过1×1Conv进行变换,来编码时间和通道方向上的特征
        x_t, x_c = torch.split(tc, [channel_group, T], dim=-1) # 重新分割: x_t: (B*G,N,1,C/G);  x_c:(B*G,N,1,T)

        ### 1×1分支和3×3分支的输出表示  ###
        x1 = group_x * x_t.sigmoid() * x_c.permute(0, 1, 3,2).sigmoid() # 通过水平方向权重和垂直方向权重调整输入,得到1×1分支的输出: (B*G,N,T,C/G)
        x2 = self.conv3x3(group_x.permute(0,3,1,2)).permute(0,2,3,1)  # 通过3×3卷积提取时间维度上的局部上下文信息: (B*G,N,T,C/G)-permute->(B*G,C/G,N,T)-conv->(B*G,C/G,N,T)-permute-->(B*G,N,T,C/G)

        ### 跨空间学习 ###
        ## 1×1分支生成通道描述符来调整3×3分支的输出
        x11 = self.softmax(self.agp(x1)) # (B*G,N,T,C/G)-agp->(B*G,N,1,C/G)
        x12 = x2.permute(0,1,3,2) # (B*G,N,T,C/G)--permute-->(B*G,N,C/G,T)
        y1 = torch.matmul(x11, x12)  #(B*G,N,1,C/G) @ (B*G,N,C/G,T)== (B*G,N,1,T)

        ## 3×3分支生成通道描述符来调整1×1分支的输出
        x21 = self.softmax(self.agp(x2)) # 对1×3分支的输出执行平均池化,然后通过softmax获得归一化后的通道描述符: (B*G,N,T,C/G)-->agp-->(B*G,N,1,C/G)
        x22 = x1.permute(0,1,3,2)  # (B*G,N,T,C/G)--permute->(B*G,N,C/G,T)
        y2 = torch.matmul(x21, x22)  # (B*G,N,1,C/G) @ (B*G,N,C/G,T) = (B*G,N,1,T)

        # 聚合两种尺度的空间位置信息, 通过sigmoid生成空间权重, 从而再次调整输入表示
        weights = (y1 + y2).reshape(B * self.groups, N, T, 1) # (B*G,N,T,1)
        weights_ = weights.sigmoid()  # 通过sigmoid生成权重表示: (B*G,N,T,1)
        out = (group_x * weights_).reshape(B,N,T,C) # 通过空间权重再次校准输入:(B*G,N,T,C/G)--reshape-->(B,N,T,C)

        return out


if __name__ == '__main__':
    # (B,N,T,C)  N:序列的个数, T:序列长度, C:通道的数量
    input=torch.randn(1,10,96,64)
    Model = EMA(channels=64, N=10)
    output=Model(input)
    print(output.shape)
