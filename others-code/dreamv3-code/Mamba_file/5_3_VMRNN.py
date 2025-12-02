import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.swin_transformer import PatchEmbed
from VMRNN_self.vmamba import VSSBlock, SS2D  # 确保从正确的模块导入 VSSBlock 和 SS2D
from typing import Optional, Callable
from functools import partial
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VSB(VSSBlock):
    def __init__(
            self,
            hidden_dim: int = 0,
            input_resolution: tuple = (224, 224),
            drop_path: float = 0,
            norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            input_resolution=input_resolution,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            **kwargs
        )
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.input_resolution = input_resolution

    def forward(self, x, hx=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.ln_1(x) # 继承VSSBlock中的LayerNorm

        if hx is not None:
            hx = self.ln_1(hx) # 使隐藏状态h_t也通过LayerNorm: (B,L,C)
            x = torch.cat((x, hx), dim=-1) # 拼接输入x和隐藏状态h_t: (B,L,2C)
            x = self.linear(x) # 恢复与输入相同的通道: (B,L,C)
        x = x.view(B, H, W, C) # 转换为2维空间表示：(B,L,C)-->(B,H,W,C)

        x = self.drop_path(self.self_attention(x)) # 继承VSSBlock中的self_attention方法  (B,H,W,C)-->(B,H,W,C)

        x = x.view(B, H * W, C) # (B,H,W,C)-view->(B,HW,C)
        x = shortcut + x

        return x


class VMRNNCell(nn.Module):
    def __init__(self, hidden_dim, input_resolution, depth,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, d_state=16, **kwargs):
        """
        Args:
        hidden_dim: Dimension of the hidden layer.
        input_resolution: Tuple of the input resolution.
        depth: Depth of the cell.
        drop, attn_drop, drop_path: Parameters for VSB.
        norm_layer: Normalization layer.
        d_state: State dimension for SS2D in VSB.
        """
        super(VMRNNCell, self).__init__()


        self.VSBs = nn.ModuleList(
            VSB(hidden_dim=hidden_dim, input_resolution=input_resolution,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, attn_drop_rate=attn_drop,
                d_state=d_state, **kwargs)
            for i in range(depth))

    def forward(self, xt, hidden_states):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C).to(xt.device)
            cx = torch.zeros(B, L, C).to(xt.device)
        else:
            # print(hidden_states)
            hx, cx = hidden_states

        outputs = []
        # 执行多个VSS Block, 在这里默认设置为1层
        for index, layer in enumerate(self.VSBs):
            if index == 0:
                x = layer(xt, hx) # xt:(B,L,C), hx:(B,L,C); x:(B,L,C)
                outputs.append(x)
            else:
                x = layer(outputs[-1], None)  # Assuming VSB does not use hx for layers after the first
                outputs.append(x)

        o_t = outputs[-1]
        Ft = torch.sigmoid(o_t) # (B,L,C)-->(B,L,C)

        cell = torch.tanh(o_t)  # (B,L,C)-->(B,L,C)

        Ct = Ft * (cx + cell) #更新cell,即c_t
        Ht = Ft * torch.tanh(Ct) # 更新隐藏状态h_T

        return Ht, (Ht, Ct)




if __name__ == '__main__':
    # (B,L,C)   B:batchsize; L:序列长度/图像patch序列; C:通道数量
    x1 = torch.randn(1,196,64).to(device)
    hidden_states = (torch.randn(1, 196, 64).to(device),torch.randn(1, 196, 64).to(device)) # (h_t, c_t)

    Model = VMRNNCell(hidden_dim=64,
                                input_resolution=(14, 14),
                                depth=1, drop=0.,
                                attn_drop=0., drop_path=0.1,
                                norm_layer=nn.LayerNorm, d_state=16).cuda()  # input_resolution在这里是指划分patch后的图像分辨率, 以patch_siz=16划分,划分完之后图像是14*14,也就是196个patch

    x,hidden_state  = Model(x1,hidden_states) # (B,L,C)-->(B,L,C)
    print(x.shape)