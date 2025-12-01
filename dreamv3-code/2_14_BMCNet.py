import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import DropPath, to_2tuple
from torch.nn import init

def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)

        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


# class BIE(nn.Module):
#     def __init__(self, nf=64):
#         super(BIE, self).__init__()
#         # self-process
#         self.conv1 = ResidualBlock_noBN(nf)
#         self.conv2 = self.conv1
#         self.convf1 = nn.Conv2d(nf * 2, nf, 1, 1, padding=0)
#         self.convf2 = self.convf1
#
#         self.scale = nf ** -0.5
#         self.norm_s = LayerNorm2d(nf)
#         self.clustering = nn.Conv2d(nf, nf, 1, 1, padding=0)
#         self.unclustering = nn.Conv2d(nf * 2, nf, 1, stride=1, padding=0)
#
#         self.v1 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)
#         self.v2 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)
#
#         # initialization
#         initialize_weights([self.convf1, self.convf2, self.clustering, self.unclustering, self.v1, self.v2], 0.1)
#
#     def forward(self, x_1, x_2, x_s):
#         b, c, h, w = x_1.shape
#
#         x_1_ = self.conv1(x_1) # (b,c,h,w)-->(b,c,h,w)
#         x_2_ = self.conv2(x_2) # (b,c,h,w)-->(b,c,h,w)
#         shared_class_center1 = self.clustering(self.norm_s(self.convf1(torch.cat([x_s, x_2], dim=1)))).view(b, c, -1) # xs,x2--concat-->(b,2c,h,w)-convf1->(b,c,h,w)--clustering->(b,c,h,w)-view->(b,c,h*w)
#         shared_class_center2 = self.clustering(self.norm_s(self.convf2(torch.cat([x_s, x_1], dim=1)))).view(b, c, -1) # xs,x1--concat-->(b,2c,h,w)-convf2->(b,c,h,w)--clustering->(b,c,h,w)-view->(b,c,h*w)
#
#         v_1 = self.v1(x_1).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]
#         v_2 = self.v2(x_2).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]
#
#         att1 = torch.bmm(shared_class_center1, v_1) * self.scale # [b, c, hw] x [b, hw, c] -> [b, c, c]
#         att2 = torch.bmm(shared_class_center2, v_2) * self.scale  # [b, c, hw] x [b, hw, c] -> [b, c, c]
#
#         out_1 = torch.bmm(torch.softmax(att1, dim=-1), v_1.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw]
#         out_2 = torch.bmm(torch.softmax(att2, dim=-1), v_2.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw]
#
#         x_s_ = self.unclustering(torch.cat([shared_class_center1.view(b, c, h, w), shared_class_center2.view(b, c, h, w)], dim=1)) + x_s
#
#         return out_1 + x_2_, out_2 + x_1_, x_s_


class BIE(nn.Module):
    def __init__(self, nf=64):
        super(BIE, self).__init__()
        # self-process
        self.conv1 = ResidualBlock_noBN(nf)
        self.conv2 = self.conv1
        self.convf1 = nn.Conv2d(nf * 2, nf, 1, 1, padding=0)
        self.convf2 = self.convf1

        self.scale = nf ** -0.5
        self.norm_s = LayerNorm2d(nf)
        self.clustering = nn.Conv2d(nf, nf, 1, 1, padding=0)
        self.unclustering = nn.Conv2d(nf * 2, nf, 1, stride=1, padding=0)

        self.v1 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)
        self.v2 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)

        self.gated1 = nn.Sequential(
            nn.Conv2d(nf, nf, 1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.gated2 = nn.Sequential(
            nn.Conv2d(nf, nf, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # initialization
        initialize_weights([self.convf1, self.convf2, self.clustering, self.unclustering, self.v1, self.v2], 0.1)

    def forward(self, x_p, x_n, x_int):
        b, c, h, w = x_p.shape

        x_p_ = self.conv1(x_p) # (b,c,h,w)-->(b,c,h,w)
        x_n_ = self.conv2(x_n) # (b,c,h,w)-->(b,c,h,w)

        # concat-conv-norm-conv
        x_xp = self.clustering(self.norm_s(self.convf1(torch.cat([x_int, x_p], dim=1)))).view(b, c, -1) # xs,x2--concat-->(b,2c,h,w)-convf1->(b,c,h,w)--clustering->(b,c,h,w)-view->(b,c,h*w)
        x_xn = self.clustering(self.norm_s(self.convf2(torch.cat([x_int, x_n], dim=1)))).view(b, c, -1) # xs,x1--concat-->(b,2c,h,w)-convf2->(b,c,h,w)--clustering->(b,c,h,w)-view->(b,c,h*w)

        # 得到key矩阵
        k_p = self.v1(x_p).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]
        k_n = self.v2(x_n).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]

        # 计算注意力矩阵
        att1 = torch.bmm(x_xp, k_n) * self.scale # [b, c, hw] x [b, hw, c] -> [b, c, c]
        att2 = torch.bmm(x_xn, k_p) * self.scale  # [b, c, hw] x [b, hw, c] -> [b, c, c]

        # 对value矩阵进行加权
        z_pn = torch.bmm(torch.softmax(att1, dim=-1), k_n.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw] -> [b,c,h,w]
        z_np = torch.bmm(torch.softmax(att2, dim=-1), k_p.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw] -> [b,c,h,w]

        # 门控融合
        W_p = self.gated1(z_pn + x_p_)
        Y_p = W_p * z_pn + (1-W_p) * x_p_

        W_n = self.gated1(z_np + x_n_)
        Y_n = W_n * z_np + (1 - W_n) * x_n_


        Y = self.unclustering(torch.cat([x_xp.view(b, c, h, w), x_xn.view(b, c, h, w)], dim=1)) + x_int

        return Y_p, Y_n, Y



if __name__ == '__main__':
    # (B,C,H,W)
    x1 = torch.randn(1, 64, 224, 224)
    x2 = torch.randn(1, 64, 224, 224)
    x3 = torch.randn(1, 64, 224, 224)

    # 定义BIE
    Model = BIE(nf=64)
    out = Model(x1,x2,x3)
    print(out[0].shape,out[1].shape,out[2].shape)
