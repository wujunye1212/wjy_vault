import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

'''
题目：Learning to Upsample by Learning to Sample
即插即用的上采样模块：DySample

我们推出了 DySample，这是一款超轻且高效的动态上采样器。
虽然最近基于内核的动态上采样器（如 CARAFE、FADE 和 SAPA）取得了令人印象深刻的性能提升，
但它们引入了大量工作负载，主要是由于耗时的动态卷积和用于生成动态内核的额外子网络。
我们实现了一个新上采样器 DySample。

该上采样适用于：语义分割、目标检测、实例分割、全景分割。
style='lp' / ‘pl’ 用该模块上采样之前弄清楚这两种风格
'''
class DySample_UP(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super(DySample_UP,self).__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())
    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        # i=self.sample(x, offset)
        # print("lp",i.shape)
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        # i=self.sample(x, offset)
        # print("pl",i.shape)
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)
    '''
    # 'lp' (局部感知):
    这种风格直接在输入特征图的每个局部区域生成偏移量，然后基于这些偏移进行上采样。
    这意味着每个输出像素的位置都是由其对应输入区域内的内容直接影响的，
    适用于需要精细控制每个输出位置如何从输入特征中取样的情况。
    在需要保持局部特征连续性和细节信息的任务（如图像超分辨率、细节增强）中，'lp' 风格可能会表现得更好。
  
    # 'pl' (像素shuffle后局部感知):
    在应用偏移量之前，首先通过像素shuffle操作打乱输入特征图的像素排列，
    这实质上是一种空间重排，能够促进通道间的信息交互。随后，再进行与'lp'类似的局部感知上采样。
    这种风格可能更有利于全局上下文的融合和特征的重新组织，适合于那些需要较强依赖于相邻区域上下文信息的任务
    （例如语义分割，全景分割）。像素shuffle增加了特征图的表征能力，有助于模型捕捉更广泛的上下文信息。
 
    # 两者各有优势，依赖于特定任务的需求：
    如果任务强调保留和增强局部细节，那么 'lp' 可能是更好的选择。
    如果任务需要更多的全局上下文信息和特征重组，'pl' 可能更合适。
    '''

if __name__ == '__main__':
    input = torch.rand(1, 64, 4, 4)
    # in_channels=64, scale=4, style='lp'/‘pl’,
    DySample_UP = DySample_UP(in_channels=64,scale=2,style='pl')
    output = DySample_UP(input)
    print('input_size:', input.size())
    print('output_size:', output.size())