# https://github.com/ndsclark/MCANet
# https://pdf.sciencedirectassets.com/271095/1-s2.0-S0952197623X00128/1-s2.0-S0952197623012630/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEP7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDt%2FEVysGctq%2B5fP1K5XD8t2NFAvDjkpeO6wqi8ed4uxAiAngdekcU19P3FjVt6s7JOgib%2F2gOtE3yctQJlDtYdc%2BSqzBQhHEAUaDDA1OTAwMzU0Njg2NSIMQoE5kMbqOINayV7mKpAFv6cigvNAENR2%2FtY%2B7GV4CHGF4PCHQoujFlZXOc%2BXbx9YmgFpes8kjK5nrPAyL6WDLXbxhbcr0FMxP%2FUFSXjDP5czTlh18GZ0KmwGotTt3J96VCDciSwX7dYnNx%2BESQp4uFIR6Pf9Njd%2B0V6nHfqStzZoo8cs0TGtMnWomMZXKo%2FTk%2BbEYjI2Qi72BQFftVjwu9wf2YQDwGO81Eu8iRlzXI2zeVzBEmh%2ByJcBMHsIUMSoU%2FCEipScLiyBe2DU0FFnzrMQTLvDYLum7MuGkF0VN%2FXINiMGT44P9S3%2FebJyXsdTGTVlGtCxW0VuQIAEFeK2vkD%2FQsSY3vF4WTVsohS3FWUUzzMLpHQ6X96y5JzvF8%2Bb7hBzwh4vFOMChJ8eGpeypodq7w2aBlq2D9ySscWajEUUkUb%2Bp%2FQb%2FcbOyaPB9%2FUlIbWJiTlIqCeInIMVu0C9RpJ70udOwlsRpFiTh8BkmcB6Eaf0260SFKFcpimH9qk9%2FosEperxl2CoVtwBvvLwVqx%2F4Z6r6U0UdxqVUvDvdoeqUA7hptCb73VkjdhgOdcP%2FkXQijLghlcjPWjq9klJiY3JSwLXmGgo%2BoB5vuin1YlagIbpI%2Fbt01KIub5jZZ3YtzHHT5cCUZihqkNkVCLNDOJjeVgMiWx1P7dd9yF79nTWysVCCr7MPHbYWxuqA5X5mbxRh6NITo6R7KRlGX7G2111F%2B5F3cq8uTJR8SccL86sJH0DB6sSmsnDVHFypbBDi61ba4oN0oXvXJtOkn6Fc1pvrU090HHm6D%2FvKLvto8snywNwhATbG9RDV3ECvnRbfs2vKrAtbwh1Vko1v9alcaovB2kQQFwuIlQDA8UeWwJ6Rgj3k3sBGrJUH029d9Iwq4ngtwY6sgFPmt5HELOCjBI3czdI0lcqLGlw7KtRcDx2t4J6y%2B1ewJ5UOGIjsDvUfDwLH0QngCIT0eiTgf9YnjeOMgisfTXLM3Eq17d6X6e0w8OexkgysA%2B7VL6NCwuD0BvjQzRLaiTrril5%2BANfSpKpDwLJkJ3AB7jfkcI0U7FNayFiyc%2FgbmHI7%2FHX0%2B%2BFLA7qe%2BKWaZdXYJHr64Pt3AJcqmVIJiONIkBHA4OVvfyXxjmpq19X5rIz&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240928T143100Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY3367JGTP%2F20240928%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=dd4e76321862f1ded1fa6ddd61ef885da13465e63f34f281e1588c4ceccda2f0&hash=8508af89f5fe8cfb75dec8b47daa96c4bef2b92d7182f7a02cbf25d8acea264d&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0952197623012630&tid=spdf-4493270a-b302-479d-a003-a89368e27d58&sid=ccee36a79dbc804b977902e3b208fa999a28gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=070e58050454550453&rr=8ca46f40cecacec9&cc=cn
import torch
from torch import nn
import math
'''
SCI二区 2023 
深度卷积神经网络（CNNs）的性能。然而，大多数现有的方法要么忽略了在通道和空间维度上的建模关注，
要么引入了更高的模型复杂度和更重的计算负担。为了缓解这一困境，在本文中，我们提出了一种轻量级和高效的多维协作注意，
MCA，这是一种利用三分支体系结构同时推断通道、高度和宽度维度的注意的新方法，几乎没有自由的额外开销。
MCA的基本组成部分，我们不仅开发一个自适应组合机制合并双跨维特征响应压缩转换，提高特征描述符的信息量和可辨别性，
还设计一个门机制激励转换自适应地决定交互的覆盖捕获局部特性交互，克服性能的悖论和计算开销的权衡。
我们的MCA简单而通用，可以很容易地插到各种经典的cnn作为即插即用模块。

本文总结：
MCA（多维协作注意力）模块，这是一种用于深度卷积神经网络（CNN）的轻量高效的注意力机制，主要用于图像识别任务。
MCA 模块通过同时在通道、宽度和高度三个维度推理注意力，提升了特征的表现力，能够显著提升网络的性能，并且引入的计算开销极小。

MCA模块的作用：
1.跨维度注意力推理：MCA 模块在通道、宽度和高度三个维度上协同推理注意力，
帮助网络更好地确定"应该注意什么"和"在何处注意"。这使得网络在进行图像识别时能够更准确地捕捉重要的特征。

2.自适应特征聚合：在 squeeze 阶段，MCA 使用全局平均池化和标准差池化来提取不同维度的特征响应，
并通过自适应组合机制，将这些特征有效结合，增强特征描述的表现力。

3.局部特征交互捕捉：在 excitation 阶段，MCA 使用自适应的方式捕捉本地特征交互，
而不使用复杂的降维策略，从而在保持高效的同时提升模型性能。

4.模块设计轻量化：MCA 采用了三分支架构，每个分支分别处理通道、宽度和高度维度，
能够很好地整合空间和通道之间的关系，并且易于集成到现有的 CNN 架构中，不会带来显著的计算负担。（即插即用模块）

适用于:图像分类，目标检测，图像分割等所有CV任务通用的即插即用注意力模块
'''



__all__ = ['MCALayer', 'MCAGate']


class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.size()

        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)

        return std


class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """Constructs a MCAGate module.
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(MCAGate, self).__init__()

        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError

        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]

        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"

        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()

        out = self.sigmoid(out)
        out = out.expand_as(x)

        return x * out


class MCALayer(nn.Module):
    def __init__(self, inp, no_spatial=True):
        """Constructs a MCA module.
        Args:
            inp: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super(MCALayer, self).__init__()

        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1

        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)

    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.c_hw(x)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)

        return x_out
# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    models = MCALayer(32)
    input = torch.rand(1, 32, 64, 64)
    output = models(input)
    print('input_size:',input.size())
    print('output_size:',output.size())