import torch
import torch.nn as nn
from timm.models.layers import DropPath
# https://github.com/Tianfang-Zhang/CAS-ViT/tree/main
# https://arxiv.org/abs/2408.03703
'''
CAS-ViT： 用于高效移动应用的卷积加法自注意力视觉变压器
即插即用模块：CAS卷积加法自注意力模块
Vision Transformers （ViTs） 凭借其令牌混合器强大的全局上下文功能，
标志着神经网络的革命性进步。然而，成对令牌亲和性和复杂的矩阵操作限制了
它在资源受限场景和实时应用程序（如移动设备）上的部署，尽管在以前的工作中
已经做出了相当大的努力。在本文中，我们介绍了 CAS-ViT：卷积加法自注意力视觉转换器，
以实现移动应用程序效率和性能之间的平衡。
首先，我们认为 token mixer 获取全局上下文信息的能力取决于多种信息交互，例如空间域和通道域。
随后，我们按照这个范式构建了一个新的加法相似函数，并提出了一种名为 Convolutional Additive Token Mixer （CATM） 的高效实现。
这种简化可显著降低计算开销。
我们评估了 CAS-ViT 在各种视觉任务中的表现，包括图像分类、目标检测、实例分割和语义分割。
进行的实验表明，与其他最先进的主干相比，CAS-ViT 实现了具有竞争力的性能，使其成为高效移动视觉应用的可行选择。

'''
class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.block(x)
class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)
class CAS(nn.Module):
    """
    改变了proj函数的输入，不对q+k卷积，而是对融合之后的结果proj
    """
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out
class Mlp_CASVIT(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class LocalIntegration(nn.Module):
    """
    """
    def __init__(self, dim, ratio=1, act_layer=nn.ReLU, norm_layer=nn.GELU):
        super().__init__()
        mid_dim = round(ratio * dim)
        self.network = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0),
            norm_layer(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1, groups=mid_dim),
            act_layer(),
            nn.Conv2d(mid_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.network(x)

class AdditiveTokenMixer(nn.Module):
    """
    改变了proj函数的输入，不对q+k卷积，而是对融合之后的结果proj
    """
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out


class AdditiveBlock(nn.Module):
    """
    """
    def __init__(self, dim, mlp_ratio=4., attn_bias=False, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.local_perception = LocalIntegration(dim, ratio=1, act_layer=act_layer, norm_layer=norm_layer)
        self.norm1 = norm_layer(dim)
        self.attn = AdditiveTokenMixer(dim, attn_bias=attn_bias, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_CASVIT(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.local_perception(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# 输入 B C H W ,  输出 B C H W
if __name__ == '__main__':
    # 创建一个输入张量，形状为 B C H W
    input = torch.randn(1, 512, 64, 64)
    # 创建 CAS 模块的实例
    # model = CAS(dim=512)  #缝合方案一
    model =  AdditiveBlock(dim=512) #缝合方案二
    # 前向传播，获取输出
    output = model(input)
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
