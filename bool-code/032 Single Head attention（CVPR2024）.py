import torch
'''
    论文地址：https://arxiv.org/pdf/2401.16456
    论文题目：SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design（CVPR 2024）
    中文题目：SHViT：高效宏设计的单头视觉Transformer
    讲解视频：https://www.bilibili.com/video/BV1zVf7YPE55/
    单头注意力模块：
        在注意力操作之前的缩小了K和V的空间尺度，这大大减少了计算/内存开销。
'''
class GroupNorm(torch.nn.GroupNorm):
    """
    使用1个组进行的Group Normalization。
    输入：形状为[B, C, H, W]的张量
    """
    def __init__(self, num_channels, **kwargs):
        # 初始化时指定1个组和传入的通道数
        super().__init__(1, num_channels, **kwargs)

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,groups=1, bn_weight_init=1):
        super().__init__()
        # 添加一个卷积层到序列中，无偏置
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        # 添加一个BatchNorm2d层到序列中
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        # 初始化BatchNorm层的权重
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        # 初始化BatchNorm层的偏差
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        # 计算融合后的卷积权重
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        # 计算融合后的卷积偏置
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        # 创建一个新的卷积层并设置参数
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class SHSA(torch.nn.Module):
    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()
        # 设置缩放因子
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim            # 默认16
        self.dim = dim

        self.pdim = pdim        # 默认32
        # 定义预归一化层
        self.pre_norm = GroupNorm(pdim)
        # 定义qkv转换层
        # self.qkv = Conv2d_BN(32, 16 * 2 + 32)  -> Conv2d_BN(32, 64)
        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        # 定义投影层
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(dim, dim, bn_weight_init=0))

    def forward(self, x):
        # x = torch.randn(1, 64, 32, 32)
        B, C, H, W = x.shape

        # 分割输入张量
        # x1 torch.Size([1, 32, 32, 32])
        # x_id torch.Size([1, 32, 32, 32])
        x1, x_id = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)

        # 对x1进行预归一化
        """
            GroupNorm 层，它将整个通道作为一个组，计算所有通道上的均值和方差，并对这些通道进行归一化。
            目的：为了确保在应用自注意力机制之前，输入数据已经被适当地标准化，以帮助模型更好地学习特征表示。
        """
        x1 = self.pre_norm(x1)  # torch.Size([1, 32, 32, 32])
        qkv = self.qkv(x1)      # torch.Size([1, 64, 32, 32])
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)# Q torch.Size([1, 16, 32, 32]) K torch.Size([1, 16, 32, 32]) V torch.Size([1, 32, 32, 32])
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # Q torch.Size([1, 16, 1024]) K torch.Size([1, 16, 1024]) V torch.Size([1, 32, 1024])
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1) # torch.Size([1, 1024, 1024])

        """
            V                      ：torch.Size([1, 32, 1024])
            attn.transpose(-2, -1) :torch.Size([1, 1024, 1024])
                            torch.Size([1, 32, 1024])
            x1                     ：torch.Size([1, 32, 32, 32])
        """
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)

        # 拼接处理后的x1和原始的x2，并通过投影层
        # x1 torch.Size([1, 32, 32, 32])
        # x_id torch.Size([1, 32, 32, 32])
        ##  ([1, 64, 32, 32])
        x = self.proj(torch.cat([x1, x_id], dim=1))
        return x


if __name__ == '__main__':
    x = torch.randn(1, 64, 32, 32)
    model = SHSA(64)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")