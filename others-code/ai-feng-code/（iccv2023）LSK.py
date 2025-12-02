import torch
import torch.nn as nn

'''
近期关于遥感物体检测的研究主要集中在改进定向边界框的表示，
但忽视了遥感场景中呈现的独特先验知识。这类先验知识很有用，
因为微小的遥感对象如果没有参考足够长程的上下文可能会被误检，而且不同类型的对象所需的长程上下文也各不相同。

在本文中，我们考虑到了这些先验知识，并提出了大型选择性核网络（LSKNet）。
LSKNet能够动态调整其大的空间接收域，以便更好地建模遥感场景中各种物体的范围上下文。
据我们所知次，这是首在遥感物体检测领域探索大型和选择性核机制。
'''
class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = LSKblock(64).cuda()
    input = torch.rand(1, 64, 64, 64).cuda()
    output = block(input)
    print(input.size(), output.size())
