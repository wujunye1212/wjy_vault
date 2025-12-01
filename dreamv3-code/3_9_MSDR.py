import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GMSDRCell(torch.nn.Module):
    def __init__(self, num_units, input_dim, num_nodes, pre_k, nonlinearity='tanh'):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_units = num_units
        self._supports = []
        self.pre_k = pre_k
        self.pre_v = 1
        self.input_dim = input_dim
        self.R = nn.Parameter(torch.zeros(pre_k, num_nodes, self._num_units), requires_grad=True)
        self.attlinear = nn.Linear(num_nodes * self._num_units, 1)
        self.iplinear = nn.Linear(2 * num_units, num_units)


    def forward(self, inputs, hx_k):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx_k: (B, pre_k, num_nodes, rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        bs, k, n, d = hx_k.shape  # hx_k:(B,K,N,D), 表示前K个时间步的信息     output:(B,ND);
        preH = hx_k[:, -1:] # 前一个时间步信息:(B,1,N,D)
        for i in range(1, self.pre_v):
            preH = torch.cat([preH, hx_k[:, -(i + 1):-i]], -1) # 与前V个时间步拼接
        preH = preH.reshape(bs, n, d * self.pre_v) # (B,V,N,D)-->(B,N,VD)==(B,N,D)  在这里,我们将V设置为1,这意味着preH只包含前一个时间信息

        inputs = torch.reshape(inputs, (bs, n, d)) # (B,ND)-->(B,N,D)
        inputs_and_state = self.iplinear(torch.cat([inputs, preH], dim=-1)) #将当前时间步输入与前1个时间步进行拼接,并通过线性层恢复通道数量:(B,N,D)-cat-(B,N,D)-->(B,N,2D); (B,N,2D)--iplinear-->(B,N,D)

        new_states = hx_k + self.R.unsqueeze(0) # hx_k:(B,K,N,D); 可学习参数R:(1,K,N,D); 为隐状态hx_k添加一个可学习参数
        output = inputs_and_state + self.attention(new_states) # 将隐藏状态馈入到注意力模块,对k个时间步进行加权求和: (B,K,N,D)--attention-->(B,N,D);  其输出与输入进行相加
        output = output.unsqueeze(1) # (B,N,D)-->(B,1,N,D)
        x = hx_k[:, 1:k] # 取前k-1个时间步信息: (B,K-1,N,D)
        hx_k = torch.cat([x, output], dim=1) # 将前k-1个时间步信息与当前时间步输出进行拼接,重新作为新的隐藏状态: (B,K-1,N,D)-cat-(B,1,N,D)-->(B,K,N,D)
        output = output.reshape(bs, n * d) # (B,1,N,D)--reshape-->(B,ND)
        return output, hx_k


    def attention(self, inputs: Tensor):
        bs, k, n, d = inputs.size() # (B,K,N,D)
        x = inputs.reshape(bs, k, -1) # (B,K,N,D)--reshape-->(B,K,ND)
        out = self.attlinear(x) # (B,K,ND)--attlinear-->(B,K,1)
        weight = F.softmax(out, dim=1) # 在K维度上执行softmax,或者每个时间步对应的权重
        outputs = (x * weight).sum(dim=1).reshape(bs, n, d) # 1)对x进行加权:(B,K,ND)*(B,K,1)==(B,K,ND);   2)进行求和:(B,K,ND)--sum-->(B,ND)   3)重新reshape:(B,ND)--reshape-->(B,N,D)
        return outputs


class EncoderModel(nn.Module):
    def __init__(self, input_dim, num_units, seq_len, num_nodes,pre_k, num_rnn_layers):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.num_units = num_units
        self.rnn_units = num_units
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.pre_k = pre_k
        self.mlp = nn.Linear(self.input_dim, self.rnn_units)
        self.num_rnn_layers = num_rnn_layers
        self.gmsdr_layers = nn.ModuleList(
            [GMSDRCell(self.num_units, self.input_dim, self.num_nodes, self.pre_k) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hx_k):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hx_k: (num_layers, batch_size, pre_k, self.num_nodes, self.rnn_units)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hx_k # shape (num_layers, batch_size, pre_k, self.num_nodes, self.rnn_units)
                 (lower indices mean lower layers)
        """
        hx_ks = []
        batch = inputs.shape[0] # (B,ND)
        x = inputs.reshape(batch, self.num_nodes, self.input_dim) # (B,ND)--reshape-->(B,N,D)
        output = self.mlp(x).view(batch, -1) # (B,N,D)--mlp-->(B,N,D)--view-->(B,ND)
        for layer_num, dcgru_layer in enumerate(self.gmsdr_layers):
            next_hidden_state, new_hx_k = dcgru_layer(output, hx_k[layer_num]) # output:(B,ND); hx_k[layer_num]:(B,K,N,D);  next_hidden_state:(B,ND); new_hx_k:(B,K,N,D);
            hx_ks.append(new_hx_k)
            output = next_hidden_state # 将output作为下一层的输出
        return output, torch.stack(hx_ks)



class GMSDRModel(nn.Module):
    def __init__(self, input_dim, num_units, seq_len, num_nodes,pre_k, num_rnn_layers):
        super().__init__()
        self.rnn_units = num_units
        self.num_rnn_layers = num_rnn_layers
        self.pre_k = pre_k
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.encoder_model = EncoderModel(input_dim, num_units, seq_len, num_nodes,pre_k, num_rnn_layers)
        self.out = nn.Linear(self.rnn_units, input_dim)

    def forward(self, inputs):
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: hx_k: (num_layers, batch_size, pre_k, num_sensor, rnn_units)
        """
        hx_k = torch.zeros(self.num_rnn_layers, inputs.shape[1], self.pre_k, self.num_nodes, self.rnn_units,
                           device=device)  # (layer,B,K,N,D)
        outputs = []
        for t in range(self.seq_len):
            output, hx_k = self.encoder_model(inputs[t], hx_k) # inputs[t]: (B,ND);  hx_k:(layer,B,K,N,D);  output:(B,ND);  hx_k:(layer,B,K,N,D)
            outputs.append(output)
        return torch.stack(outputs), hx_k  # torch.stack(outputs): (T,B,ND)


if __name__ == '__main__':
    # (B,T,N,D) B:batchsize, T:timestep  N:number of series  D:number of channel
    x1 = torch.randn(1,96,10,64).to(device)
    B,T,N,D = x1.size()
    x1 = x1.reshape(T,B,N*D)

    Model = GMSDRModel(input_dim=D, num_units=D, seq_len=T, num_nodes=N, pre_k=4, num_rnn_layers=1).to(device)
    out,hx_k = Model(x1) # out:  (T,B,ND)
    out = out.reshape(B,T,N,D)

    print(out.shape)