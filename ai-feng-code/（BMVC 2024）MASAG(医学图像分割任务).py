import torch
import torch.nn as nn
import torch.nn.functional as F
# https://github.com/xmindflow/MSA-2Net
# https://arxiv.org/abs/2407.21640

'''
BMVC 2024
即插即用模块：MASAG  本文用于医学图像分割 适用于所有cv任务

医学图像分割涉及识别和分离医学图像中的对象实例，以描绘各种组织和结构，由于这些特征的大小、形状和密度的显著变化，
这项任务变得复杂。卷积神经网络 （CNN） 传统上用于此任务，但在捕获远程依赖关系方面存在局限性。
配备自注意力机制的变压器旨在解决这个问题。
然而，在医学图像分割中，合并局部和全局特征以有效地集成各种尺度的特征图是有益的，
既捕获了详细特征，又捕获了更广泛的语义元素，以处理结构的变化。
在本文中，我们介绍了 MSA2Net，一种新的深度分段框架，具有跳过连接的权宜之计设计。
这些连接通过动态加权并将粗粒度编码器特征与细粒度解码器特征映射相结合来促进特征融合。
具体来说，我们提出了一个多尺度自适应空间注意力 （MASAG），它可以动态调整感受野（局部和全局上下文信息），
以确保有选择地突出空间相关的特征，同时最大限度地减少背景干扰。涉及皮肤病学和放射学数据集的广泛评估表明，
我们的 MSA2Net 优于最先进的 （SOTA） 作品或与其性能相当。

'''

def num_trainable_params(model):
    nums = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return nums

class GlobalExtraction(nn.Module):
  def __init__(self,dim = None):
    super().__init__()
    self.avgpool = self.globalavgchannelpool
    self.maxpool = self.globalmaxchannelpool
    self.proj = nn.Sequential(
        nn.Conv2d(2, 1, 1,1),
        nn.BatchNorm2d(1)
    )
  def globalavgchannelpool(self, x):
    x = x.mean(1, keepdim = True)
    return x

  def globalmaxchannelpool(self, x):
    x = x.max(dim = 1, keepdim=True)[0]
    return x

  def forward(self, x):
    x_ = x.clone()
    x = self.avgpool(x)
    x2 = self.maxpool(x_)

    cat = torch.cat((x,x2), dim = 1)

    proj = self.proj(cat)
    return proj

class ContextExtraction(nn.Module):
  def __init__(self, dim, reduction = None):
    super().__init__()
    self.reduction = 1 if reduction == None else 2

    self.dconv = self.DepthWiseConv2dx2(dim)
    self.proj = self.Proj(dim)

  def DepthWiseConv2dx2(self, dim):
    dconv = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 1,
              groups = dim),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True),
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 2,
              dilation = 2),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True)
    )
    return dconv

  def Proj(self, dim):
    proj = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim //self.reduction,
              kernel_size = 1
              ),
        nn.BatchNorm2d(num_features = dim//self.reduction)
    )
    return proj
  def forward(self,x):
    x = self.dconv(x)
    x = self.proj(x)
    return x

class MultiscaleFusion(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.local= ContextExtraction(dim)
    self.global_ = GlobalExtraction()
    self.bn = nn.BatchNorm2d(num_features=dim)

  def forward(self, x, g,):
    x = self.local(x)
    g = self.global_(g)

    fuse = self.bn(x + g)
    return fuse


class MASAG(nn.Module):
    # Version 1
  def __init__(self, dim):
    super().__init__()
    self.multi = MultiscaleFusion(dim)
    self.selection = nn.Conv2d(dim, 2,1)
    self.proj = nn.Conv2d(dim, dim,1)
    self.bn = nn.BatchNorm2d(dim)
    self.bn_2 = nn.BatchNorm2d(dim)
    self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=dim, out_channels=dim,
                  kernel_size=1, stride=1))
  def forward(self,x,g):
    x_ = x.clone()
    g_ = g.clone()
    #stacked = torch.stack((x_, g_), dim = 1) # B, 2, C, H, W
    multi = self.multi(x, g) # B, C, H, W
    ### Option 2 ###
    multi = self.selection(multi) # B, num_path, H, W

    attention_weights = F.softmax(multi, dim=1)  # Shape: [B, 2, H, W]
    #attention_weights = torch.sigmoid(multi)
    A, B = attention_weights.split(1, dim=1)  # Each will have shape [B, 1, H, W]

    x_att = A.expand_as(x_) * x_  # Using expand_as to match the channel dimensions
    g_att = B.expand_as(g_) * g_
    x_att = x_att + x_
    g_att = g_att + g_
    ## Bidirectional Interaction
    x_sig = torch.sigmoid(x_att)
    g_att_2 = x_sig * g_att
    g_sig = torch.sigmoid(g_att)
    x_att_2 = g_sig * x_att
    interaction = x_att_2 * g_att_2
    projected = torch.sigmoid(self.bn(self.proj(interaction)))
    weighted = projected * x_
    y = self.conv_block(weighted)
    #y = self.bn_2(weighted + y)
    y = self.bn_2(y)
    return y

if __name__ == "__main__":
    # 创建一个简单的输入特征图
    input1 = torch.randn(1, 64, 32, 32)
    input2 = torch.randn(1, 64, 32, 32)

    # 创建一个MASAG实例
    MASAG = MASAG(dim=64)

    # 将两个输入特征图传递给 MSGA 模块
    output = MASAG(input1, input2)
    # 打印输入和输出的尺寸
    print(f"input 1 shape: {input1.shape}")
    print(f"input 2 shape: {input2.shape}")
    print(f"output shape: {output.shape}")