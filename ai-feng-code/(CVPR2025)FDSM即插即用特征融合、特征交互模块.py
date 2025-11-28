import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import to_2tuple
from torch.nn import functional as F

# DynamicFilter来自：AAAI 2024 DFFormer
# DynamicFilter来自论文： https://arxiv.org/pdf/2303.03932v2
#看Ai缝合怪b站视频：2025.6.11更新的视频
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias
def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_heads = origin_weight.shape[0:3]  # size, w, c, 2
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
    new_weight = torch.nn.functional.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)
    return new_weight
class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()     #来自Ai缝合怪复现整理
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)
        # 来自Ai缝合怪复现整理
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
class FrequencyDynamicSelection(nn.Module): #原论文名字：DynamicFilter
    def __init__(self, dim, expansion_ratio=1, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=64, weight_resize=True,
                 **kwargs):     #来自Ai缝合怪复现整理
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x,dw):
        B, H, W, _ = x.shape

        routeing = self.reweight(dw.mean(dim=(1, 2))).view(B, self.num_filters,-1).softmax(dim=1)

        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.weight_resize:     #来自Ai缝合怪复现整理
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1],
                                                    x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        #来自Ai缝合怪复现整理
        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
        # 来自Ai缝合怪复现整理
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)
        return x

class FDSM(nn.Module):
    def __init__(self, in_channels):
        super(FDSM, self).__init__()
        # RGB and NIR feature extraction
        self.rgb_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.nir_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.cpcs = nn.Sequential(
                            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
                            nn.PReLU(),
                            nn.Conv2d(in_channels, in_channels , kernel_size=3, padding=1),
                            nn.SiLU())

        # Aggregation layers
        self.mlp = ConvMlp(in_channels )
        self.softmax = nn.Softmax(dim=1)

        # FDS 频率动态选择
        self.fds =FrequencyDynamicSelection(in_channels)

    def forward(self, rgb, nir):
        b,c,h,w = rgb.shape
        rgb_feat = self.rgb_conv(rgb)
        nir_feat = self.nir_conv(nir)

        frn = self.cpcs(torch.cat([rgb_feat,nir_feat],dim=1))
        att = F.adaptive_avg_pool2d(frn, output_size=h)  # B, C, h,w
        Aggregation = self.softmax(self.mlp(att))

        rgb_feat =rgb_feat.permute(0,2,3,1) # B C H W -> B H W C
        nir_feat = nir_feat.permute(0,2,3,1)
        Aggregation = Aggregation.permute(0,2,3,1)

        feat_r = self.fds(rgb_feat, Aggregation)
        feat_n = self.fds(nir_feat,Aggregation)
        # return feat_r, feat_n
        return feat_r+feat_n

#二次创新FDAM，频率动态注意力混合模块，冲SCI三区和四区，CCF-B/C

# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input1= torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 32, 64, 64)
    # 创建 FDSM 模块
    fdsm = FDSM(in_channels=32)
    # 将输入图像传入FDSM 模块进行处理
    output = fdsm(input1,input2).permute(0,3,1,2)
    # 输出结果的形状
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新-FDSM_input_size:', input1.size())
    print('Ai缝合即插即用模块永久更新-FDSM_output_size:', output.size())

    # 创建 FDAM 模块  二次创新FDAM，频率动态注意力混合模块，冲SCI三区和四区，CCF-B/C
    # fdam = FDAM(in_channels=32)
    # # 将输入图像传入FDAM 模块进行处理
    # output = fdam(input1,input2)
    # print('顶会顶刊二次创新模块永久更新-FDAM_input_size:', input1.size())
    # print('顶会顶刊二次创新模块永久更新-FDAM_output_size:', output.size())
    #CVPR2025 FDSM模块的二次创新，FDAM在我的二次创新模块改进交流群，可以直接去发小论文！
    #二次创新模块只更新二次创新交流，永久更新中
    #二次创新改进商品链接在视频评论区