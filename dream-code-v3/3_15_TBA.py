import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        # print("input channel: ", input_dims)
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        # print("before x:", x.shape)
        for conv in self.convs:
            x = conv(x)
        # print("after x: ", x.shape)
        return x


class MAB(nn.Module):
    def __init__(self, K,d,input_dim,output_dim,bn_decay):
        super(MAB, self).__init__()
        D=K*d
        self.K = K
        self.d=d
        self.FC_q = FC(input_dims=input_dim, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=input_dim, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=input_dim, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=output_dim, activations=F.relu,
                     bn_decay=bn_decay)

    # 以第一阶段注意力为例进行注释解析
    def forward(self, Q, K, batch_size, type="spatial", mask=None):

        # 线性层变换
        query = self.FC_q(Q) # (B,R,N,D)-->(B,R,N,D)
        key = self.FC_k(K) # (B,T,N,D)-->(B,T,N,D)
        value = self.FC_v(K) # (B,T,N,D)-->(B,T,N,D)

        # 划分多头表示
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0) # (B,R,N,D)-->(B*h,R,N,d)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0) # (B,T,N,D)-->(B*h,T,N,d)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0) # (B,T,N,D)-->(B*h,T,N,d)

        if mask==None:
            if type=="temporal":
                query = query.permute(0, 2, 1, 3) # (B*h,R,N,d)-->(B*h,N,R,d)
                key = key.permute(0, 2, 1, 3) # (B*h,T,N,d)-->(B*h,N,T,d)
                value = value.permute(0, 2, 1, 3) # (B*h,T,N,d)-->(B*h,N,T,d)
            attention = torch.matmul(query, key.transpose(2, 3)) # 矩阵点积, 得到参考点对所有时间步的注意力分数: (B*h,N,R,d) @ (B*h,N,d,T) == (B*h,N,R,T)
            attention /= (self.d ** 0.5) # 缩放
            attention = F.softmax(attention, dim=-1) # softmax归一化
            result = torch.matmul(attention, value) # (B*h,N,R,T) @ (B*h,N,T,d) == (B*h,N,R,d)
            if type=="temporal":
                result = result.permute(0, 2, 1, 3) # (B*h,N,R,d)-->(B*h,R,N,d)
            result = torch.cat(torch.split(result, batch_size, dim=0), dim=-1)  # (B*h,R,N,d)-->(B,R,N,D)
            result = self.FC(result) # (B,R,N,D)-->(B,R,N,D)
        else:
            mask=torch.cat(torch.split(mask, self.K, dim=-1), dim=0)
            if type=="temporal":
                query = query.permute(0, 2, 1, 3)
                key = key.permute(0, 2, 1, 3)
                value = value.permute(0, 2, 1, 3)
                mask=mask.permute(0,2,1,3)
            if mask.shape==query.shape:
                set_mask=torch.ones_like(key).cuda()
                mask = torch.matmul(mask,set_mask.transpose(2,3))
            elif mask.shape==key.shape:
                set_mask=torch.ones_like(query).cuda()
                mask = torch.matmul(set_mask,mask.transpose(2,3))
            attention = torch.matmul(query, key.transpose(2, 3))
            attention /= (self.d ** 0.5)
            attention=attention.masked_fill(mask==0,-1e9)
            attention = F.softmax(attention, dim=-1)
            result = torch.matmul(attention, value)
            if type=="temporal":
                result = result.permute(0, 2, 1, 3)
            result = torch.cat(torch.split(result, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size
            result = self.FC(result)
        return result


class temporalAttention(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''
    def __init__(self, K, d,num_of_vertices,set_dim, bn_decay):
        super(temporalAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.num_of_vertices=num_of_vertices
        self.set_dim = set_dim
        self.I = nn.Parameter(torch.Tensor(1,set_dim, self.num_of_vertices, D))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(K, d, D, D, bn_decay)
        self.mab1 = MAB(K, d, D, D, bn_decay)

    def forward(self, X, mask):

        batch_size = X.shape[0] # B

        I = self.I.repeat(X.size(0), 1, 1, 1) # 将参考点特征重复Batchsize次: (1,R,N,D)-repeat->(B,R,N,D)

        H = self.mab0(I, X, batch_size, "temporal", mask) # 执行第一阶段注意力, 参考点作为Q, X作为K/V矩阵, 目的是将所有时间步信息聚合到参考点中: I: (B,R,N,D);  X: (B,T,N,D); H: (B,R,N,D);

        result = self.mab1(X, H, batch_size, "temporal", mask) # 执行第二阶段注意力, X作为Q, 更新后的参考点H作为K/V矩阵, 目的是将参考点信息广播到原始时间序列的每一个时间步中: H: (B,R,N,D);  X: (B,T,N,D);  result: (B,T,N,D);

        return result


if __name__ == '__main__':
    #  (B,T,N,D)
    x1 = torch.randn(10,96,10,64).to(device)
    B, T, N, D = x1.size() # T是序列长度, N是序列个数
    k = 8 # 注意力头个数
    d = int(D//k) # 每个注意力头的通道数量

    Model = temporalAttention(K=k, d=d, num_of_vertices=N, set_dim=5, bn_decay=0.1).to(device)

    out = Model(x1, mask=None) # (B,T,N,C)--> (B,T,N,C)
    print(out.shape)