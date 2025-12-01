import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x) # 3×3Conv: (B,C,H,W)--conv_0-->(B,D,H,W)
        x = self.act(x) # (B,D,H,W)-GELU->(B,D,H,W)
        x = self.conv_1(x) # 1×1Conv: (B,D,H,W)-->(B,C,H,W)
        return x


class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim,dim*2,1,1,0)
        self.linear_1 = nn.Conv2d(dim,dim,1,1,0)
        self.linear_2 = nn.Conv2d(dim,dim,1,1,0)

        self.lde = DMlp(dim,2)

        self.dw_conv = nn.Conv2d(dim,dim,3,1,1,groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1,dim,1,1)))
        self.belt = nn.Parameter(torch.zeros((1,dim,1,1)))

    def forward(self, f):
        _,_,h,w = f.shape
        # 首先通过1×1Conv,将通道(B,C,H,W)--linear-->(B,2C,H,W)--chunk-->(B,C,H,W) and (B,C,H,W)
        y, x = self.linear_0(f).chunk(2, dim=1)

        """EASA"""
        # 将输入依次通过maxpool, dwcon, 分别提取低频分量以及非局部结构信息: (B,C,H,W)--pool-->(B,C,H/8,W/8)--dwconv-->(B,C,H/8,W/8)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        # 计算x的方差: (B,C,H,W)--var-->(B,C,1,1)
        x_v = torch.var(x, dim=(-2,-1), keepdim=True)
        # 首先将非局部特征x_s与方差x_v相加, 然后通过1×1Conv进行融合,得到调制特征; 请注意:作者在相加的过程中,为两者各自乘了一个可学习的参数进行调整。
        # x_s_v: (B,C,H/8,W/8)
        x_s_v = self.gelu(self.linear_1(x_s * self.alpha.to(f.device) + x_v * self.belt.to(f.device)))
        # 使用调制特征x_s_v来聚合输入特征x, 但需要将x_s_v插值恢复为(B,C,H,W)
        x_l = x * F.interpolate(x_s_v, size=(h, w), mode='nearest')
        #x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha.to(f.device) + x_v * self.belt.to(f.device))), size=(h,w), mode='nearest')

        """LDE"""
        y_d = self.lde(y) # (B,C,H,W)--lde-->(B,C,H,W)

        return self.linear_2(x_l + y_d) # 将EASA和LDE的输出进行融合: (B,C,H,W)


if __name__ == '__main__':
    # (B,C,H,W)
    x1 = torch.randn(1, 64, 224, 224)

    # 定义SMFA
    Model = SMFA(dim=64)
    # 执行SMFA
    out = Model(x1)
    print(out.shape)