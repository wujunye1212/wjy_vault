import torch
import torch.nn as nn
import torch.nn.functional as F
# https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html 在这个网站找到与你环境相匹配的mmcv按照命令
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from einops import rearrange
import warnings
import numbers

# pip install -U openmim
# mim install mmcv==2.0.0

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm1 = LayerNorm(dim, LayerNorm_type) #如果看不懂上面的几个bias函数,或者觉得很繁琐,可以把这一行替换为普通的layerNorm函数,上面的直接删除就可以了。
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)

    def forward(self, x):
        b, c, h, w = x.shape # (B,C2,H/4,W/4)
        x1 = self.norm1(x) # (B,C2,H/4,W/4)-->(B,C2,H/4,W/4)

        # 分别在x轴和y轴上提取多尺度特征
        attn_00 = self.conv0_1(x1) # x轴上的1×7conv:(B,C2,H/4,W/4)-->(B,C2,H/4,W/4)
        attn_01 = self.conv0_2(x1) # y轴上的7×1conv:(B,C2,H/4,W/4)-->(B,C2,H/4,W/4)
        attn_10 = self.conv1_1(x1) # x轴上的1×11conv:(B,C2,H/4,W/4)-->(B,C2,H/4,W/4)
        attn_11 = self.conv1_2(x1) # y轴上的11×1conv:(B,C2,H/4,W/4)-->(B,C2,H/4,W/4)
        attn_20 = self.conv2_1(x1) # x轴上的1×21conv:(B,C2,H/4,W/4)-->(B,C2,H/4,W/4)
        attn_21 = self.conv2_2(x1) # y轴上的21×1conv:(B,C2,H/4,W/4)-->(B,C2,H/4,W/4)

        # 分别融合x轴和y轴上的多尺度特征
        out1 = attn_00 + attn_10 + attn_20 # 将x轴上的三个尺度的卷积后的特征进行相加：(B,C2,H/4,W/4) = (B,C2,H/4,W/4) + (B,C2,H/4,W/4) + (B,C2,H/4,W/4)
        out2 = attn_01 + attn_11 + attn_21 # 将y轴上的三个尺度的卷积后的特征进行相加：(B,C2,H/4,W/4) = (B,C2,H/4,W/4) + (B,C2,H/4,W/4) + (B,C2,H/4,W/4)
        out1 = self.project_out(out1) # 对x轴相加的特征通过1×1Conv进行融合：(B,C2,H/4,W/4)
        out2 = self.project_out(out2) # 对y轴相加的特征通过1×1Conv进行融合,这里x轴和y轴共享一个1×1conv层：(B,C2,H/4,W/4)

        # 分别通过x轴和y轴上的融合后的多尺度特征，来生成qkv
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads) # (B,C2,H/4,W/4)-->(B,k,H/4,d*(W/4)) C2=k*d, k是注意力头的个数，d是每个注意力头的通道数
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads) # (B,C2,H/4,W/4)-->(B,k,H/4,d*(W/4))
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads) # (B,C2,H/4,W/4)-->(B,k,W/4,d*(H/4))
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads) # (B,C2,H/4,W/4)-->(B,k,W/4,d*(H/4))
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads) # (B,C2,H/4,W/4)-->(B,k,W/4,d*(H/4))
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads) # (B,C2,H/4,W/4)-->(B,k,H/4,d*(W/4))
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        # 交叉轴注意力：q来自y轴的特征,k/v来自x轴特征
        attn1 = (q1 @ k1.transpose(-2, -1)) # 计算注意力矩阵：(B,k,H/4,d*(W/4)) @ (B,k,d*(W/4),H/4) = (B,k,H/4,H/4)
        attn1 = attn1.softmax(dim=-1) # softmax归一化注意力矩阵：(B,k,H/4,H/4)-->(B,k,H/4,H/4)
        out3 = (attn1 @ v1) + q1 # 对v矩阵进行加权：(B,k,H/4,H/4) @ (B,k,H/4,d*(W/4)) = (B,k,H/4,d*(W/4))

        # 交叉轴注意力：q来自x轴的特征,k/v来自y轴特征
        attn2 = (q2 @ k2.transpose(-2, -1)) # 计算注意力矩阵：(B,k,W/4,d*(H/4)) @ (B,k,d*(H/4),W/4) = (B,k,W/4,W/4)
        attn2 = attn2.softmax(dim=-1) #  softmax归一化注意力矩阵：(B,k,W/4,W/4)-->(B,k,W/4,W/4)
        out4 = (attn2 @ v2) + q2 # 对v矩阵进行加权：(B,k,W/4,W/4) @ (B,k,W/4,d*(H/4)) = (B,k,W/4,d*(H/4))

        # 将x和y轴上的注意力输出变换为与输入相同的shape
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w) #(B,k,H/4,d*(W/4))-->(B,C2,H/4,W/4)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w) #(B,k,W/4,d*(H/4))-->(B,C2,H/4,W/4)

        # 对x和y轴注意力的输出进行融合,并添加残差连接，shape保持不变
        out = self.project_out(out3) + self.project_out(out4) + x

        return out


class MCAHead(nn.Module):
    def __init__(self, in_channels, image_size, heads, c1_channels,
                 **kwargs):
        super().__init__()
        self.image_size = image_size
        self.decoder_level = Attention(in_channels[1], heads, LayerNorm_type='WithBias')
        self.conv_cfg = None
        self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.act_cfg = dict(type='ReLU')
        self.align = ConvModule(
            in_channels[3],
            in_channels[0],
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.squeeze = ConvModule(
            sum((in_channels[1], in_channels[2], in_channels[3])),
            in_channels[1],
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels[1] + in_channels[0],
                in_channels[3],
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                in_channels[3],
                in_channels[3],
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, inputs):
        """Forward function."""
        # inputs = self._transform_inputs(inputs)
        inputs = [resize(
            level,
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        ) for level in inputs] #将X2/X3/X4进行上采样,与X1保持相同的特征图大小,但是通道没有变化
        y1 = torch.cat([inputs[1], inputs[2], inputs[3]], dim=1) # 将X2/X3/X4进行拼接：(B,C2+C3+C4,H/4,W/4)
        x = self.squeeze(y1) # 将拼接后的多个特征进行降维,和X2保持相同的通道数量：(B,C2+C3+C4,H/4,W/4)-->(B,C2,H/4,W/4)
        x = self.decoder_level(x) # 执行交叉轴注意力：(B,C2,H/4,W/4)-->(B,C2,H/4,W/4)
        x = torch.cat([x, inputs[0]], dim=1) # 将交叉轴注意力的输出与X1进行拼接,这是为了能够更好的利用X1的边界信息,因为X1的特征图尺寸最大：(B,C2,H/4,W/4)-concat-(B,C1,H/4,W/4) == (B,C1+C2,H/4,W/4)
        x = self.sep_bottleneck(x) # 两层卷积：(B,C1+C2,H/4,W/4)-->(B,C4,H/4,W/4)-->(B,C4,H/4,W/4)
        output = self.align(x) # 恢复和X1相同的shape: (B,C4,H/4,W/4)-->(B,C1,H/4,W/4)

        return output


if __name__ == '__main__':
    # (B,C,H,W)  H和W分表示图片的尺寸, 在这里我们定义尺寸为224, X1/X2/X3/X4是四个不同尺度的特征图
    # 当然你也可以把X1/X2/X3/X4换成你所需的shape,但是要记住：这个模块的输出是和X1的shape保持一致的,所以最好把X1设置为原始特征,X2/X3/X4是X1的其他分辨率特征

    X1 = torch.randn(1,32,56,56)  # H/4 = W/4 = 56    (B,C1,H/4,W/4)
    X2 = torch.randn(1,64,28,28)  # H/8 = W/8 = 28    (B,C2,H/8,W/8)
    X3 = torch.randn(1,160,14,14) # H/16 = W/16 = 14  (B,C3,H/16,W/16)
    X4 = torch.randn(1,256,7,7)   # H/32 = W/32 = 7   (B,C4,H/32,W/32)
    inputs = [X1,X2,X3,X4]

    # 这里的image_size应该与X1中的尺寸保持相同; in_channels是四个特征的通道; head是注意力头个数; c1_channels这个特征没用,随便设置
    Model = MCAHead(in_channels=[32,64,160,256], image_size=56, heads=8, c1_channels=48)
    out = Model(inputs)
    print(out.shape)