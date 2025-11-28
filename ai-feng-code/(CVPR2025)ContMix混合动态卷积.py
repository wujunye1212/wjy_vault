import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum
from natten.functional import na2d_av # 需要安装这个模块 ，目前只支持Linux系统上安装使用
from timm.models.layers import to_2tuple

# pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
# pip install natten==0.17.1+torch230cu121 -f https://shi-labs.com/natten/wheels/
'''
来自CVPR 2025顶会卷积模块  

即插即用模块： ContMix 上下文混合卷积模块   
两个二次创新模块: DCAM 动态卷积注意力混合模块           编故事：突出局部和全局上下文特征的表征学习
               DCAFusion 动态卷积注意力特征融合模块   编故事：1.可以突出局部和全局上下文特征的表征学习，
                                                       2.高频特征与低频特征融合的优势（分析一下高频特征存在的缺点和优点，低频特征的缺点和优点。）
                                                         二者强强联合的优点 
                                                      
动态卷积与上下文混合模块的主要目的是在保持强局部归纳偏置的同时，增强卷积神经网络(ConvNet)对长程依赖关系的建模能力。
这种能力对于提高模型在图像分类和语义分割等任务中的表现尤其重要。
ContMix原理：
1.亲和度计算：首先，基于输入特征图X，通过特定变换生成Q（查询）和K（键），其中Q代表了原始输入的转换结果，
         而K是通过自适应平均池化将X聚合到多个区域中心的结果。然后，将Q和K分成若干组，每组分别计算其间的亲和度，形成一系列亲和度矩阵。

2.动态核生成：每个亲和度矩阵的每一行经过一个可学习的线性层转换后，生成空间变化的动态卷积核。这些卷积核包含了来自自上而下的语义上下文的全局信息。
        当这些动态生成的卷积核应用于特征图时，每个token都与内置于核中的全局信息交互，从而有效捕捉长程依赖关系，同时保留对局部结构的敏感性。

3.性能优化：该模块通过上述机制克服了传统卷积操作中固定大小卷积核导致的有限感受野问题，
        同时也解决了现有动态卷积方法无法同时具备全局建模能力和强局部感知的问题。

总结: ContMix模块通过引入动态卷积核并结合上下文信息，使得ConvNet能够更好地捕捉输入图像中的长程依赖关系，从而提升了模型在各种视觉任务中的性能表现。


ContMix和2个创新模块（DCAM和DCAFusion）模块适用于：
   特征提取、在图像分类、语义分割、遥感语义分割、图像分割、目标检测、图像增强、图像恢复、低光照增强、超分辨率图像等所有CV任务通用即插即用模块


'''
def get_conv2d(in_channels,
               out_channels,
               kernel_size,
               stride,
               padding,
               dilation,
               groups,
               bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (
    kernel_size[0] // 2, kernel_size[1] // 2)

    if attempt_use_lk_impl and need_large_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print(
                '---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)

    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (
                conv_bias - bn.running_mean) * bn.weight / std


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """

    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size // 2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                     padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0), bias=True,
                                     attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class ContMix(nn.Module):
    def __init__(self,
                 dim=64,
                 kernel_size=7,
                 smk_size=5,
                 num_heads=2,
                 deploy=False,
                 use_gemm=False
                 ):
        super().__init__()
        ctx_dim = dim // 2
        self.kernel_size = kernel_size
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False,
                                attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
        )

        self.weight_query = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=False),  # 32 -> 32
            nn.BatchNorm2d(dim // 2),
        )

        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(ctx_dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2),
        )

        self.weight_proj = nn.Conv2d(49, kernel_size ** 2 + smk_size ** 2, kernel_size=1)

        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.get_rpb()

    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1))
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2))
        nn.init.zeros_(self.rpb1)
        nn.init.zeros_(self.rpb2)

    @torch.no_grad()
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return (idx_h, idx_w, idx_k)

    def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h[kernel_size // 2] = height - (kernel_size - 1)
        num_repeat_w[kernel_size // 2] = width - (kernel_size - 1)
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (
                    2 * kernel_size - 1)) + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size ** 2))
        bias_idx = torch.flip(bias_idx, [0])
        rpb = torch.flatten(rpb, 1, 2)[:, bias_idx]
        rpb = rpb.reshape(1, int(self.num_heads), int(height), int(width), int(kernel_size ** 2))
        return attn + rpb

    def forward(self, x):
        B, C, H, W = x.shape
        lepe = self.lepe(x)

        query, key = torch.chunk(x, 2, dim=1)
        query = self.weight_query(query) * self.scale
        key = self.weight_key(key)
        query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        key = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        weight = einsum(query, key, 'b g c n, b g c l -> b g n l')
        weight = rearrange(weight, 'b g n l -> b l g n').contiguous()
        weight = self.weight_proj(weight)
        weight = rearrange(weight, 'b l g (h w) -> b g h w l', h=H, w=W)

        attn1, attn2 = torch.split(weight, split_size_or_sections=[self.smk_size ** 2, self.kernel_size ** 2], dim=-1)
        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)
        attn1 = self.apply_rpb(attn1, self.rpb1, H, W, self.smk_size, *rpb1_idx)
        attn2 = self.apply_rpb(attn2, self.rpb2, H, W, self.kernel_size, *rpb2_idx)
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)
        value = rearrange(x, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)

        x1 = na2d_av(attn1, value[0], kernel_size=self.smk_size)
        x2 = na2d_av(attn2, value[1], kernel_size=self.kernel_size)

        x = torch.cat([x1, x2], dim=1)
        x = rearrange(x, 'b g h w c -> b (g c) h w', h=H, w=W)
        x = self.dyconv_proj(x)


        x = x + lepe
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
'''二次创新模块：DCAM 动态卷积注意力混合模块    '''
class DCAM(nn.Module):
    def __init__(self, dim,kernel_size=7, num_heads=4, bias=False):
        super(DCAM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True, groups=dim // self.num_heads, padding=1)
        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=False, use_sync_bn=False,
                                attempt_use_lk_impl=False),
            nn.BatchNorm2d(dim),
        )
    def forward(self, x):
        b, c, h, w = x.shape
        lepe = self.lepe(x)
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)

        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + lepe
        return output
'''二次创新模块：DCAFusion动态卷积注意力特征融合模块    '''
class DCAFusion(nn.Module):
    def __init__(self, dim):
        super(DCAFusion, self).__init__()
        self.DCAM = DCAM(dim)
        self.PixelAttention = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        pattn1 = self.DCAM(initial)
        pattn2 = self.sigmoid(self.PixelAttention(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result
if __name__ == "__main__":
    input = torch.randn(1, 64, 32, 32)
    model = ContMix(dim=64)    # 初始化 ContMix 模块  只能在Linux上运行，因为需要安装natten包
    output = model(input)    # 执行前向传播
    print("ContMix-input_size:", input.shape)
    print("ContMix-output_size:", output.shape)

    DCAMmodel = DCAM(dim=64)    # 初始化 DCAM 动态卷积注意力混合模块模块
    output = DCAMmodel(input)    # 执行前向传播
    print("二次创新—DCAM_input_size:", input.shape)
    print("二次创新—DCAM_output_size:", output.shape)

    DCAFusion = DCAFusion(dim=64)  # 初始化 DCAFusion动态卷积注意力特征融合模块
    input1 = torch.randn(1, 64, 32, 32)
    input2 = torch.randn(1, 64, 32, 32)
    output = DCAFusion(input1,input2)  # 执行前向传播
    print("二次创新—DCAFusion_input_size:", input.shape)
    print("二次创新—DCAFusion_output_size:", output.shape)
