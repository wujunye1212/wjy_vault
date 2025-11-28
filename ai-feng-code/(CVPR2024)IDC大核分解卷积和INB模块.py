import torch
import torch.nn as nn
from timm.layers import to_2tuple
from timm.models.layers import DropPath

'''
来自CVPR 2024 顶会
即插即用模块： InceptionDWConv2d（IDC）和InceptionNeXtBlock （INB）

InceptionNeXt 模块的提出是为了解决在现代卷积神经网络（CNN）中使用大核卷积时，效率与性能之间的矛盾问题。
受到 Vision Transformer (ViT) 长距离建模能力的启发，许多研究表明大核卷积可以扩大感受野并提高模型性能。
例如，ConvNeXt 使用 7×7 深度卷积来提升效果。然而，尽管大核卷积在理论上 FLOPs 较少，
但其高内存访问成本导致在高性能计算设备上效率下降。并通过实验表明大核卷积在计算效率方面存在问题。

InceptionNeXt 模块通过引入高效的大核卷积分解技术，解决了大核卷积在计算效率上的问题。
其主要作用包括：
1.卷积分解：将大核卷积分解为多个并行的分支，其中包含小核卷积、带状核（1x11 和 11x1）以及身份映射，
使模型能够高效地利用大感受野，同时减少计算开销。
2.提高计算效率：通过分解卷积核来提升计算效率，减少大核卷积带来的高计算成本，实现速度与性能的平衡。
3.扩大感受野：带状核能够在保持较低计算成本的情况下扩大感受野，从而捕捉更多的空间信息。
4.性能优势：在不牺牲模型性能的前提下，InceptionNeXt 模块提高了推理速度，尤其适合高性能与高效率需求的场景。
InceptionNeXt 模块通过分解大核卷积的创新设计，在保持模型准确率的同时，显著提升了推理速度。

'''
class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class InceptionNeXtBlock(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer=InceptionDWConv2d,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,

    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

# 输入 B C H W   输出 B C H W
if __name__ == '__main__':
    # 创建输入张量
    input = torch.randn(1, 32, 64,64)

    INB = InceptionNeXtBlock(32)
    IDC = InceptionDWConv2d(32)

    output = INB(input)
    print('INB_input_size:',input.size())
    print('INB_output_size:',output.size())

    output = INB(input)
    print('IDC_input_size:', input.size())
    print('IDC_output_size:', output.size())
