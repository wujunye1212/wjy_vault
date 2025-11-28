import torch
import torch.nn as nn
from typing import Tuple, Optional, List
'''xLSTM改进
xLSTM（高级版）之所以称之为xLSTM就是因为它将LSTM扩展为多个LSTM的变体，sLSTM(初级版)和mLSTM（中级版），
每种变体都针对特定的性能和功能进行优化，以处理各种复杂的序列数据问题。

xLSTM之sLSTM（初级版slstm）
sLSTM（ScalarLSTM）在传统的LSTM基础上增加了标量更新机制。这种设计通过对内部记忆单元进行细粒度的控制，
优化了门控机制，使其更适合处理有着细微时间变化的序列数据。sLSTM通常会使用指数门控和归一化技术，
以改善模型在长序列数据处理上的稳定性和准确性。通过这种方式，sLSTM能够在保持较低计算复杂度的同时，
提供与复杂模型相当的性能，特别适用于资源受限的环境或需要快速响应的应用。
'''
class sLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        # Store the input and hidden size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Combine the Weights and Recurrent weights into a single matrix
        self.W = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.randn(self.input_size + self.hidden_size, 4 * self.hidden_size)
            ),
            requires_grad=True,
        )
        # Combine the Bias into a single matrix
        if self.bias:
            self.B = nn.Parameter(
                (torch.zeros(4 * self.hidden_size)), requires_grad=True
            )

    def forward(
        self,
        x: torch.Tensor,
        internal_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        # Unpack the internal state
        h, c, n, m = internal_state  # (batch_size, hidden_size)

        # Combine the weights and the input
        combined = torch.cat((x, h), dim=1)  # (batch_size, input_size + hidden_size)
        # Calculate the linear transformation
        gates = torch.matmul(combined, self.W)  # (batch_size, 4 * hidden_size)

        # Add the bias if included
        if self.bias:
            gates += self.B

        # Split the gates into the input, forget, output and stabilization gates
        z_tilda, i_tilda, f_tilda, o_tilda = torch.split(gates, self.hidden_size, dim=1)

        # Calculate the activation of the states
        z_t = torch.tanh(z_tilda)  # (batch_size, hidden_size)
        # Exponential activation of the input gate
        i_t = torch.exp(i_tilda)  # (batch_size, hidden_size)
        # Exponential activation of the forget gate
        f_t = torch.sigmoid(f_tilda)  # (batch_size, hidden_size)

        # Sigmoid activation of the output gate
        o_t = torch.sigmoid(o_tilda)  # (batch_size, input_size)
        # Calculate the stabilization state
        m_t = torch.max(torch.log(f_t) + m, torch.log(i_t))  # (batch_size, hidden_size)
        # Calculate the input stabilization state
        i_prime = torch.exp(i_tilda - m_t)  # (batch_size, hidden_size)

        # Calculate the new internal states
        c_t = f_t * c + i_prime * z_t  # (batch_size, hidden_size)
        n_t = f_t * n + i_prime  # (batch_size, hidden_size)

        # Calculate the stabilized hidden state
        h_tilda = c_t / n_t  # (batch_size, hidden_size)

        # Calculate the new hidden state
        h_t = o_t * h_tilda  # (batch_size, hidden_size)
        return h_t, (
            h_t,
            c_t,
            n_t,
            m_t,
        )  # (batch_size, hidden_size), (batch_size, hidden_size), (batch_size, hidden_size), (batch_size, hidden_size)
    #python版本<=3.8
    def init_hidden(self, batch_size: int, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # python版本>3.8
    # def init_hidden(self, batch_size: int, **kwargs ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
        )

class sLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        self.cells = nn.ModuleList(
            [
                sLSTMCell(input_size if layer == 0 else hidden_size, hidden_size, bias)
                for layer in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        # Permute the input tensor if batch_first is True
        if self.batch_first:
            x = x.permute(1, 0, 2)

        # Initialize the hidden states if not provided
        if hidden_states is None:
            hidden_states = self.init_hidden(x.size(1), device=x.device, dtype=x.dtype)
        else:
            # Check if the hidden states are of the correct length
            if len(hidden_states) != self.num_layers:
                raise ValueError(
                    f"Expected hidden states of length {self.num_layers}, but got {len(hidden_states)}"
                )
            if any(state[0].size(0) != x.size(1) for state in hidden_states):
                raise ValueError(
                    f"Expected hidden states of batch size {x.size(1)}, but got {hidden_states[0][0].size(0)}"
                )

        H, C, N, M = [], [], [], []

        for layer, cell in enumerate(self.cells):
            lh, lc, ln, lm = [], [], [], []
            for t in range(x.size(0)):
                h_t, hidden_states[layer] = (
                    cell(x[t], hidden_states[layer])
                    if layer == 0
                    else cell(H[layer - 1][t], hidden_states[layer])
                )
                lh.append(h_t)
                lc.append(hidden_states[layer][0])
                ln.append(hidden_states[layer][1])
                lm.append(hidden_states[layer][2])

            H.append(torch.stack(lh, dim=0))
            C.append(torch.stack(lc, dim=0))
            N.append(torch.stack(ln, dim=0))
            M.append(torch.stack(lm, dim=0))

        H = torch.stack(H, dim=0)
        C = torch.stack(C, dim=0)
        N = torch.stack(N, dim=0)
        M = torch.stack(M, dim=0)

        return H[-1], (H, C, N, M)

    def init_hidden(
        self, batch_size: int, **kwargs
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:

        return [cell.init_hidden(batch_size, **kwargs) for cell in self.cells]



if __name__ == '__main__':
    # 定义输入张量的参数
    input_dim = 128
    hidden_size = 128
    num_layers = 2
    # 初始化 sLSTM 模块
    slstm = sLSTM(input_dim, hidden_size, num_layers)#要求输入3维度张量: B,L,C
    # 随机生成输入4维度张量：B, C, H, W
    input_img = torch.randn(8,input_dim,16,16)
    input = input_img
    input_img = input_img.reshape(8,input_dim,-1).transpose(-1,-2)
    # 运行前向传递
    output, hidden_state = slstm(input_img)
    output =output.view(8,input_dim,16,16)#将三维度转化成图片四维度张量
    # 输出输入图片张量和输出图片张量的形状
    print("sLSTM_input size:", input.size())
    print("sLSTM_Output size:", output.size())
