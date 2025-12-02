import torch
import torch.nn as nn
import torch.distributions as td

#代码：https://github.com/aL3x-O-o-Hung/CSAM/blob/main/CSAM_modules.py
#论文：https://openaccess.thecvf.com/content/WACV2024/papers/Hung_CSAM_A_2.5D_Cross-Slice_Attention_Module_for_Anisotropic_Volumetric_Medical_WACV_2024_paper.pdf
'''
            分析一下这篇WACV2024顶会论文的摘要

大量的体积医学数据，尤其是磁共振成像（MRI）数据，通常是各向异性的，
因为 切片间 的分辨率通常远低于 切片内 的分辨率。  ---交代了论文的研究背景（文章肯定跟医学相关），
                                            描述了MRI数据集的特点：各向异性，分辨率（切片间2D<切片内3D）

基于深度学习的3D和纯2D分割方法在处理这类体积数据时都有所不足。         --- 之前的深度学习方法在这类体积数据集上存在不足
3D方法在面对各向异性数据时性能会受影响，而2D方法则忽略了重要的体积信息。 --- 3D方法--各向异性影响，2D方法缺乏了这种体积信息

尽管有一些研究涉及2.5D方法，但其中主要是结合体积信息使用2D卷积。        --- 现有的2.5D，学习切片之间，缺点：参数量大
这些模型侧重于学习切片之间的关系，但通常需要大量参数进行训练。

我们提出了一种跨切片注意力模块（CSAM），该模块具有极少的可训练参数，         
并通过在不同尺度的深度特征图上应用语义、位置和切片注意力，捕捉整个体积中的信息。 --- 本文的2.5D方法
                                                                  创新点：交叉切片注意力模块（CSAM)
                                                                 优点：参数量比以往方法小,考虑到了体积中的信息

我们在使用不同网络架构和任务的广泛实验中证明了CSAM的有效性和通用性。 ---通过一些实验，对比实验，消融试验。验证我们2.5D方法发好
'''

def custom_max(x,dim,keepdim=True):
    temp_x=x
    for i in dim:
        temp_x=torch.max(temp_x,dim=i,keepdim=True)[0]
    if not keepdim:
        temp_x=temp_x.squeeze()
    return temp_x

class PositionalAttentionModule(nn.Module):
    def __init__(self):
        super(PositionalAttentionModule,self).__init__()
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(7,7),padding=3)
    def forward(self,x):
        max_x=custom_max(x,dim=(0,1),keepdim=True)
        avg_x=torch.mean(x,dim=(0,1),keepdim=True)
        att=torch.cat((max_x,avg_x),dim=1)
        att=self.conv(att)
        att=torch.sigmoid(att)
        return x*att

class SemanticAttentionModule(nn.Module):
    def __init__(self,in_features,reduction_rate=16):
        super(SemanticAttentionModule,self).__init__()
        self.linear=[]
        self.linear.append(nn.Linear(in_features=in_features,out_features=in_features//reduction_rate))
        self.linear.append(nn.ReLU())
        self.linear.append(nn.Linear(in_features=in_features//reduction_rate,out_features=in_features))
        self.linear=nn.Sequential(*self.linear)
    def forward(self,x):
        max_x=custom_max(x,dim=(0,2,3),keepdim=False).unsqueeze(0)
        avg_x=torch.mean(x,dim=(0,2,3),keepdim=False).unsqueeze(0)
        max_x=self.linear(max_x)
        avg_x=self.linear(avg_x)
        att=max_x+avg_x
        att=torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1)
        return x*att

class SliceAttentionModule(nn.Module):
    def __init__(self,in_features,rate=4,uncertainty=True,rank=5):
        super(SliceAttentionModule,self).__init__()
        self.uncertainty=uncertainty
        self.rank=rank
        self.linear=[]
        self.linear.append(nn.Linear(in_features=in_features,out_features=int(in_features*rate)))
        self.linear.append(nn.ReLU())
        self.linear.append(nn.Linear(in_features=int(in_features*rate),out_features=in_features))
        self.linear=nn.Sequential(*self.linear)
        if uncertainty:
            self.non_linear=nn.ReLU()
            self.mean=nn.Linear(in_features=in_features,out_features=in_features)
            self.log_diag=nn.Linear(in_features=in_features,out_features=in_features)
            self.factor=nn.Linear(in_features=in_features,out_features=in_features*rank)
    def forward(self,x):
        max_x=custom_max(x,dim=(1,2,3),keepdim=False).unsqueeze(0)
        avg_x=torch.mean(x,dim=(1,2,3),keepdim=False).unsqueeze(0)
        max_x=self.linear(max_x)
        avg_x=self.linear(avg_x)
        att=max_x+avg_x
        if self.uncertainty:
            temp=self.non_linear(att)
            mean=self.mean(temp)
            diag=self.log_diag(temp).exp()
            factor=self.factor(temp)
            factor=factor.view(1,-1,self.rank)
            dist=td.LowRankMultivariateNormal(loc=mean,cov_factor=factor,cov_diag=diag)
            att=dist.sample()
        att=torch.sigmoid(att).squeeze().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x*att


class CSAM(nn.Module):
    def __init__(self,num_slices,num_channels,semantic=True,positional=True,slice=True,uncertainty=True,rank=5):
        super(CSAM,self).__init__()
        self.semantic=semantic
        self.positional=positional
        self.slice=slice
        if semantic:
            self.semantic_att=SemanticAttentionModule(num_channels)
        if positional:
            self.positional_att=PositionalAttentionModule()
        if slice:
            self.slice_att=SliceAttentionModule(num_slices,uncertainty=uncertainty,rank=rank)
    def forward(self,x):
        if self.semantic:
            x=self.semantic_att(x)
        if self.positional:
            x=self.positional_att(x)
        if self.slice:
            x=self.slice_att(x)
        return x
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    models = CSAM(num_slices=10,num_channels=64).cuda()
    input = torch.randn(10, 64, 128, 128).cuda()
    output = models(input)
    print('input_size:',input.size())
    print('output_size:',output.size())