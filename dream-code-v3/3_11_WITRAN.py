import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WITRAN_2DPSGMU_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, water_rows, water_cols, res_mode='none'):
        super(WITRAN_2DPSGMU_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.water_rows = water_rows
        self.water_cols = water_cols
        self.res_mode = res_mode
        # parameter of row cell
        self.W_first_layer = torch.nn.Parameter(torch.empty(6 * hidden_size, input_size + 2 * hidden_size)) # (6C,3C)
        self.W_other_layer = torch.nn.Parameter(torch.empty(num_layers - 1, 6 * hidden_size, 4 * hidden_size)) # (layer,6C,4C)
        self.B = torch.nn.Parameter(torch.empty(num_layers, 6 * hidden_size)) # (layer,6C)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def linear(self, input, weight, bias, batch_size, slice, Water2sea_slice_num):
        a = F.linear(input, weight) # input: (WB,3C); weight:(6C,3C); a:(WB,6C)
        if slice < Water2sea_slice_num:
            a[:batch_size * (slice + 1), :] = a[:batch_size * (slice + 1), :] + bias # 添加偏置
        return a

    def forward(self, input, batch_size, input_size, flag):
        if flag == 1: # cols > rows 等价于 H>W
            input = input.permute(2, 0, 1, 3)  # (B,W,H,C)-->(H,B,W,C)
        else:  # cols < rows 等价于 H<W
            input = input.permute(1, 0, 2, 3)  # 以此为例: (B,W,H,C)-->(W,B,H,C)
        Water2sea_slice_num, _, Original_slice_len, _ = input.shape # (W,B,H,C); Water2sea_slice_num:W的长度, Original_slice_len:H的长度
        Water2sea_slice_len = Water2sea_slice_num + Original_slice_len - 1  # Water2sea_slice_len: W+H-1, 它将用于循环的次数
        hidden_slice_row = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device) # 初始化行隐藏状态: (WB,C)
        hidden_slice_col = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device) # 初始化列隐藏状态: (WB,C)
        input_transfer = torch.zeros(Water2sea_slice_num, batch_size, Water2sea_slice_len, input_size).to(input.device) #存储输入数据: (W,B,W+H-1, C)

        # 在W方向上循环, r∈[0,W-1], 使W方向上的每一个样本都对应H个垂直方向上的值
        for r in range(Water2sea_slice_num):
            # input_transfer: (W,B,W+H-1, C);  input:(W,B,H,C);  将input的每一个样本(B,H,C)放入到input_transfer;  对于input_transfer来说,W方向上的每个样本在第2个维度上都有H个数值
            # input_transfer: (W,B,W+H-1, C);   0:H, 1:H+1, 2:H+2, ..., W-1:W+H-1;  大家看解析里面的RAN结构,对于每一行,都有H个值,但是坐标是逐渐递进的
            # 我们再简化一下, 忽略掉batchsize和通道C, 那么就只剩下(W,W+H-1),如《手把手带你发论文》中阐述的RAN架构那样,是一个W行,(W+H-1)列的特征图, 但是有的位置是有时间步特征的,有的位置是用0填充的
            input_transfer[r, :, r:r+Original_slice_len, :] = input[r, :, :, :]

        hidden_row_all_list = []
        hidden_col_all_list = []
        for layer in range(self.num_layers):
            if layer == 0:
                a = input_transfer.reshape(Water2sea_slice_num * batch_size, Water2sea_slice_len, input_size) # (W,B,W+H-1,C)--> (WB,W+H-1,C)
                W = self.W_first_layer  # 用于第一层的线性变换: (6C,3C)
            else:
                a = F.dropout(output_all_slice, self.dropout, self.training) # 对前一层的输出应用dropout
                if layer == 1:
                    layer0_output = a
                W = self.W_other_layer[layer-1, :, :] # 后续层的线性变换:(6C,4C)
                hidden_slice_row = hidden_slice_row * 0 # 行方向的隐藏向量:(WB,C)
                hidden_slice_col = hidden_slice_col * 0 # 列方向的隐藏向量:(WB,C)
            B = self.B[layer, :]  # 偏置: (6C),计算门控值时加到线性变换结果上

            # 对于每个时间步, 都会处理行和列上的隐藏状态,更新并产生属于输出, slice∈[0,W+H-2]
            output_all_slice_list = []
            for slice in range (Water2sea_slice_len): # W+H-1
                # gate generate;  先将行、列隐藏状态与(当前列包含的所有时间步数据)进行拼接:(WB,C)-cat-(WB,C)-cat-(WB,C)==(WB,3C); 再通过Linear:(WB,3C)--Linear-->(WB,6C)
                # a:(WB,W+H-1,C),存储输入数据;  a[:, slice, :]: (WB,C),表示取出第slice列的数据
                gate = self.linear(torch.cat([hidden_slice_row, hidden_slice_col, a[:, slice, :]],
                    dim = -1), W, B, batch_size, slice, Water2sea_slice_num)
                # gate
                sigmod_gate, tanh_gate = torch.split(gate, 4 * self.hidden_size, dim = -1) #将gate分成两部分,后续分别通过sigmoid和tanh: (WB,6C)--split-->(WB,4C) and (WB,2C)
                sigmod_gate = torch.sigmoid(sigmod_gate) # 通过sigmoid函数: (WB,4C)
                tanh_gate = torch.tanh(tanh_gate) # 通过tanh函数: (WB,2C)
                update_gate_row, output_gate_row, update_gate_col, output_gate_col = sigmod_gate.chunk(4, dim = -1) # 将sigmod_gate分成四份,行和列单元中分别有更新门和输出门:(WB,4C)--chunk-->(WB,C),(WB,C),(WB,C),(WB,C)
                input_gate_row, input_gate_col = tanh_gate.chunk(2, dim = -1) # 将tanh_gate分成两份,行和列单元中分别有输入门: (WB,2C)--chunk-->(WB,C) and (WB,C)
                # gate effect
                hidden_slice_row = torch.tanh(
                    (1-update_gate_row)*hidden_slice_row + update_gate_row*input_gate_row) * output_gate_row #通过门控机制更新行方向的隐藏状态:(WB,C)
                hidden_slice_col = torch.tanh(
                    (1-update_gate_col)*hidden_slice_col + update_gate_col*input_gate_col) * output_gate_col # 通过门控机制更新列方向的隐藏状态:(WB,C)
                # output generate
                output_slice = torch.cat([hidden_slice_row, hidden_slice_col], dim = -1) # 将更新后的行、列隐藏状态拼接生成最终输出: (WB,C)-cat-(WB,C)-->(WB,2C)
                # save output
                output_all_slice_list.append(output_slice) # 将当前循环的输出添加到列表中: (WB,2C)

                # 保存每一行的隐藏状态, slice∈[0,W+H-2], 如果当前slice大于等于H-1,即slice的范围是【H-1,W+H-2】,长度为W; Original_slice_len==H
                if slice >= Original_slice_len - 1:
                    # 如果当前slice==H-1, 即need_save_row_loc=0,第一行所有时间步计算完成; 如果slice==W+H-2, 即need_save_row_loc=W-1, 第W行所有时间步计算完成; 因此,即need_save_row_loc的范围是【0,W-1】,长度为W
                    need_save_row_loc = slice - Original_slice_len + 1
                    hidden_row_all_list.append(
                        hidden_slice_row[need_save_row_loc*batch_size:(need_save_row_loc+1)*batch_size, :]) # hidden_slice_row:(WB,C);  总共将W个时间步(代表W行)的隐藏状态依次放入列表: [0:B,C],[B:2B,C],...,[(W-1)B:WB,C]; 换句话说,依次把第一行/第二行/第三行/.../第W行的最终隐藏状态放入列表;  再换句话说,在保存行隐藏状态的时候,(WB,C)看作(B,W,C),依次(按照索引0/1/2/3)在W维度上选择对应切片(B,C)进行输出

                # 保存每一列的隐藏状态, slice∈[0,W+H-2], 如果使slice大于等于W-1,即slice的范围是【W-1,W+H-2】,长度为H;  Water2sea_slice_num=W
                if slice >= Water2sea_slice_num - 1:
                    hidden_col_all_list.append(
                        hidden_slice_col[(Water2sea_slice_num-1)*batch_size:, :]) # hidden_slice_col:(WB,C);  总共将H个(H列)隐藏状态[(W-1)B:WB,C]依次放入列表;  也就是说,在保存列隐藏状态的时候,每次选择的最后一行的向量进行保存: (WB,C)看作(B,W,C),也就是在W维度上选择最后一个切片(B,C)进行输出
                # hidden transfer
                hidden_slice_col = torch.roll(hidden_slice_col, shifts=batch_size, dims = 0) #在每个时间步都会进行滚动,确保隐藏状态在列方向上能够传播

            if self.res_mode == 'layer_res' and layer >= 1: # layer-res
                output_all_slice = torch.stack(output_all_slice_list, dim = 1) + layer0_output # 添加残差连接 [0:B,C]
            else:
                output_all_slice = torch.stack(output_all_slice_list, dim = 1) # 不添加残差连接,  (W+H-1)个(WB,2C)进行堆叠, 得到(WB,W+H-1,2C)

        hidden_row_all = torch.stack(hidden_row_all_list, dim = 1) # 将W个(B,C)进行堆叠: [B,W,C]
        hidden_col_all = torch.stack(hidden_col_all_list, dim = 1) # 将H个(B,C)进行堆叠: [B,H,C]
        hidden_row_all = hidden_row_all.reshape(batch_size, self.num_layers, Water2sea_slice_num, hidden_row_all.shape[-1]) # [B,W,C]->(B,1,W,C); layer_num=1
        hidden_col_all = hidden_col_all.reshape(batch_size, self.num_layers, Original_slice_len, hidden_col_all.shape[-1]) # [B,H,C]-->(B,1,H,C); layer_num=1
        if flag == 1:
            return output_all_slice, hidden_col_all, hidden_row_all
        else:
            return output_all_slice, hidden_row_all, hidden_col_all


if __name__ == '__main__':

    # batch_size, seq_len, input_size  (B,T,C)
    x1 = torch.randn(10, 20, 64).to(device)
    # (batch_size, water_rows, water_cols, input_size)  (B,W,H,C); 行使用W表示,列使用H表示; input_size在这里表示通道C.
    x1 = x1.reshape(10,4,5,64) # 将长度为20的1D序列重塑为一个四行五列的2D序列
    B,W,H,C = x1.size() # 用于传参
    num_layers = 1 # 定义层数

    # 接受6个参数
    Model = WITRAN_2DPSGMU_Encoder(input_size=C, hidden_size=C, num_layers=num_layers, dropout=0., water_rows=W, water_cols=H).to(device)

    # output_all_slice:保存了(W+H-1)个的行和列隐藏状态的拼接;   (WB,W+H-1,2C)
    # enc_hid_row: W行隐藏状态的拼接(共有num_layers层);    (B,layer,W,C)
    # enc_hid_col: H列隐藏状态的拼接(共有num_layers层);    (B,layer,H,C)
    output_all_slice, enc_hid_row, enc_hid_col = Model(x1, batch_size=B, input_size=C, flag=0)

    # 输出的shape: (WB,W+H-1,2C), (B,layer,W,C), (B,layer,H,C)
    print(output_all_slice.shape,enc_hid_row.shape,enc_hid_col.shape)


    # 在得到W和H方向的输出之后,将其进行融合. 这部分大家可以按照自己的方式来设计
    FC = nn.Linear(num_layers * (W + H) * C, W*H*C).to(device) # 定义一个线性层
    hidden_all = torch.cat([enc_hid_row, enc_hid_col], dim=2) # 在W和H方向上拼接: (B,layer,W,C)--cat--(B,layer,H,C)-->(B,layer,W+H,C)
    hidden_all = hidden_all.reshape(hidden_all.shape[0], -1) # (B,layer,W+H,C)--reshape-->(B,layer*(W+H)*C)
    last_output = FC(hidden_all) # (B,layer*(W+H)*C)--linear-->(B,W*H*C)
    last_output = last_output.reshape(last_output.shape[0], H*W, -1) # 恢复与输入相同的shape(B,W*H*C)--reshape-->(B,WH,C)

    print(last_output.shape)

