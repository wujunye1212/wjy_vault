import torch
import torch.nn as nn

# 主页：https://space.bilibili.com/346680886
# 代码讲解：https://www.bilibili.com/video/BV1aBxse5EEr/
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k(nn.Module):
    """CSP Bottleneck with customizable kernel sizes."""
    # https://arxiv.org/abs/1911.11929
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions and optional C3k blocks."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)

        # 创建模块列表 m，其中每个元素都是 C3k 或者 Bottleneck 实例
        self.m = nn.ModuleList(
            C3k(self.c_, self.c_, 2, shortcut, g) if c3k else Bottleneck(self.c_, self.c_, shortcut, g) for _ in
            range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))  # 将 cv1 的输出分成两部分
        y.extend(m(y[-1]) for m in self.m)  # 对每个块应用模块列表中的操作
        return self.cv2(torch.cat(y, 1))  # 将所有输出连接起来，并通过 cv2

if __name__ == '__main__':

    c3k2_module = C3k2(c1=64, c2=128, n=2, c3k=True)  # 创建一个 C3k2 模块
    input_tensor = torch.randn(1, 64, 416, 416)  # 创建一个随机输入张量
    output_tensor = c3k2_module(input_tensor)  # 将输入张量传递给 C3k2 模块

    print(input_tensor.size())
    print(output_tensor.size())
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
