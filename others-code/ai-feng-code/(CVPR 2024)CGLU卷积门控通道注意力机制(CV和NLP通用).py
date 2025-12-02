import torch
import torch.nn as nn
# https://github.com/DaiShiResearch/TransNeXt/tree/main
# https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_TransNeXt_Robust_Foveal_Visual_Perception_for_Vision_Transformers_CVPR_2024_paper.pdf

# CVPR 2024
'''
TransNeXt：用于视觉转换器的鲁棒中央凹视觉感知

由于残差连接中的深度降解效应，许多依赖堆叠层进行信息交换的
高效视觉转换器模型往往无法形成足够的信息混合，导致视觉感知不自然。

为了解决这个问题，在本文中，我们提出了CGLU --- 卷积门控通道注意力机制

门控通道注意力机制门控线性单元 （GLU） 是一种通道混频器，
已被证明在各种自然语言处理任务中性能优于多层感知器 （MLP）。
GLU 由两个线性投影组成，这两个线性投影按元素相乘，
其中一个投影由门控函数激活。

我们发现，只需在GLU的门控分支的激活函数之前添加一个最小形式的3×3深度卷积，
就可以使其结构符合门控通道注意力的设计理念，并将其转化为基于最近邻特征的门控通道注意力机制。
我们将此方法命名为卷积 GLU。CGLU
'''
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class CGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
if __name__ == '__main__':
    models = CGLU(in_features=768, hidden_features=512, out_features=768)

    # 1.如何输入的是图片4维数据 . CV方向的小伙伴都可以拿去使用
    # 随机生成输入4维度张量：B, C, H, W
    input_img = torch.randn(2, 768, 14, 14)
    input = input_img
    input_img = input_img.reshape(2, 768, -1).transpose(-1, -2)
    # 运行前向传递
    output = models(input_img,14,14)
    output = output.view(2, 768, 14, 14)  # 将三维度转化成图片四维度张量
    # 输出输入图片张量和输出图片张量的形状
    print("CV_CGLU_input size:", input.size())
    print("CV_CGLU_Output size:", output.size())

    # 2.如何输入的3维数据 . NLP方向的小伙伴都可以拿去使用
    B, N, C = 2, 196, 768  # 批量大小、序列长度、特征维度
    H, W = 14, 14  # 重塑后的高度和宽度
    input = torch.randn(B, N, C)
    output = models(input,H,W)
    print('NLP_CGLU_size:',input.size())
    print('NLP_CGLU_size:',output.size())

