import torch
from thop import profile, clever_format
from torch import nn
from torch.cuda.amp import autocast
from vision_lstm import ViLBlock, SequenceTraversal

'''
在医学图像处理领域，精准的图像分割技术对于疾病的诊断、治疗规划和研究至关重要。
近年来，深度学习方法在这一领域取得了显著进展，其中卷积神经网络（CNNs）和视觉变换器（ViTs）
因其强大的局部特征提取能力和全局上下文捕捉能力而备受青睐。然而，这些方法在处理长距离依赖关系时仍面临挑战，
尤其是在面对高分辨率或高维成像模式时。  

为了应对医学图像分割中的挑战，近期的研究提出了整合具有长距离依赖性的计算模块，
这些模块在序列长度上展现出线性的计算和内存复杂度。在这些计算模块中，
状态空间模型（State Space Models, SSMs）如Mamba已经证明了其巨大的成功，
并且已经成功地集成到了传统的UNet架构中。

研究人员仅将经过Mamba改进的UNet（U-Mamba）中的Mamba模块替换为xLSTM作为其骨干网络。
实验结果惊奇的发现，仅将Mamba模块替换成xLSTM，就可以在2D和3D医学图像分割上获得不错的性能提升，
通过广泛的实验验证，xLSTM-UNet在多个2D和3D的医学图像分割数据集上的表现均超越了现有的基于CNN、基于Transformer以及基于Mamba的分割网络。
'''
class vision_xLSTM(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        # print(f"ViLLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.vil = ViLBlock(
            dim=self.dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT
        )
        self.channel_token = channel_token  ## whether to use channel as tokens

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_vil = self.vil(x_flat)
        out = x_vil.transpose(-1, -2).reshape(B, d_model, *img_dims)

        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_vil = self.vil(x_flat)
        out = x_vil.reshape(B, n_tokens, *img_dims)

        return out

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)

        return out

if __name__ == '__main__':
    input = torch.randn(1,32,64,64)
    model = vision_xLSTM(32)
    output = model(input)
    print("vision_xLSTM_input size:", input.size())
    print("vision_xLSTM_Output size:", output.size())
