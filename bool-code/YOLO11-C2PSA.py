import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块

# 主页：https://space.bilibili.com/346680886
# 代码讲解：https://www.bilibili.com/video/BV1aBxse5EEr/
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """自动填充以保持输出形状与输入相同。"""
    if d > 1:
        # 如果有膨胀率，计算实际的卷积核大小
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # 如果没有指定填充，则根据卷积核大小自动计算填充
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p  # 返回计算后的填充值

class Conv(nn.Module):
    """标准卷积层，参数包括输入通道数、输出通道数、卷积核大小、步长、填充、组数、膨胀率和激活函数。"""
    default_act = nn.SiLU()  # 默认激活函数为 SiLU

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # 卷积层
        self.bn = nn.BatchNorm2d(c2)
        # 批归一化层
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # 激活函数，默认为 SiLU，也可以是其他激活函数或无激活函数

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
        # 前向传播：先进行卷积，然后批归一化，最后应用激活函数

    def forward_fuse(self, x):
        return self.act(self.conv(x))
        # 融合模式下的前向传播：直接进行卷积并应用激活函数

class Attention(nn.Module):
    """注意力机制模块，用于对输入张量进行自注意力操作。"""
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads  # 注意力头的数量
        self.head_dim = dim // num_heads  # 每个注意力头的维度
        self.key_dim = int(self.head_dim * attn_ratio)  # 注意力键的维度
        self.scale = self.key_dim**-0.5  # 注意力分数的缩放因子
        nh_kd = self.key_dim * num_heads  # 总的注意力键维度
        h = dim + nh_kd * 2  # 计算 qkv 的总维度
        self.qkv = Conv(dim, h, 1, act=False)  # 生成查询、键和值的卷积层
        self.proj = Conv(dim, dim, 1, act=False)  # 投影卷积层
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)  # 位置编码卷积层

    def forward(self, x):
        B, C, H, W = x.shape  # 获取输入张量的形状
        N = H * W  # 输入张量的空间尺寸

        qkv = self.qkv(x)  # 生成查询、键和值
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )  # 分割 qkv

        attn = (q.transpose(-2, -1) @ k) * self.scale  # 计算注意力分数
        attn = attn.softmax(dim=-1)  # 对注意力分数进行 softmax 归一化
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))  # 应用注意力并加上位置编码
        x = self.proj(x)  # 投影到原始维度
        return x  # 返回处理后的张量

class PSABlock(nn.Module):
    """PSA 块，包含多头注意力机制和前馈神经网络。"""
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)  # 注意力模块
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))  # 前馈神经网络
        self.add = shortcut  # 是否使用残差连接

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)  # 应用注意力机制，并可选地加上残差连接
        x = x + self.ffn(x) if self.add else self.ffn(x)  # 应用前馈神经网络，并可选地加上残差连接
        return x  # 返回处理后的张量

class C2PSA(nn.Module):
    """C2PSA 模块，包含多个 PSA 块。"""
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2  # 确保输入和输出通道数相同
        self.c = int(c1 * e)  # 隐藏层通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 第一个卷积层
        self.cv2 = Conv(2 * self.c, c1, 1)  # 第二个卷积层
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
        # 创建 n 个 PSABlock 组成的序列

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)  # 将 cv1 的输出分成两部分
        b = self.m(b)  # 对第二部分应用 PSABlock 序列
        return self.cv2(torch.cat((a, b), 1))  # 将两部分拼接并通过第二个卷积层

# 调用 C2PSA 模块的示例
if __name__ == '__main__':
    # 创建 C2PSA 实例
    c2psa_module = C2PSA(c1=256, c2=256, n=3, e=0.5)  # 3 个 PSABlock 层，扩展比为 0.5

    # 创建一个随机输入张量
    input_tensor = torch.randn(1, 256, 64, 64)

    # 将输入张量传递给 C2PSA 模块
    output_tensor = c2psa_module(input_tensor)

    # 打印输入和输出张量的形状
    print("输入张量的形状:", input_tensor.size())
    print("输出张量的形状:", output_tensor.size())

    # 输出调试信息
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")