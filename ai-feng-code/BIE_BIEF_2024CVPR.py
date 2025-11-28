import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

'''
来自CVPR 2024顶会论文 
即插即用特征融合模块： BIE 双边信息交互模块 （特征融合模块）
含二次创新 BIEF 双边信息有效融合模块 效果优于BIE，这个模块非常适合自由发挥，冲一下SCI一区 

本文提出了一种双边事件挖掘与互补网络（ BMCNet），旨在充分挖掘每类事件的潜力，
同时捕获共享信息以实现互补。具体而言，我们采用双流网络分别对每种事件进行全面挖掘。
为了促进两条流之间的信息交互，我们提出了一种双边信息交换（BIE）模块，
该模块逐层嵌入到两条流之间，有效地传播分层全局信息，同时减轻事件固有特性带来的无效信息的影响。

适用于:目标检测，图像分割，语义分割，暗光增强，图像去噪，图像增强等所有计算机视觉CV任务通用模块
'''
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

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)
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
class BIE_3(nn.Module):
    def __init__(self, nf=64):
        super(BIE_3, self).__init__()
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

        # initialization
        initialize_weights([self.convf1, self.convf2, self.clustering, self.unclustering, self.v1, self.v2], 0.1)

    def forward(self,x_1, x_2, x_s ):
        # x_1, x_2, x_s =x
        b, c, h, w = x_1.shape

        x_1_ = self.conv1(x_1)
        x_2_ = self.conv2(x_2)
        shared_class_center1 = self.clustering(self.norm_s(self.convf1(torch.cat([x_s, x_2], dim=1)))).view(b, c, -1) # [b, c, h, w] -> [b, c, h*w]
        shared_class_center2 = self.clustering(self.norm_s(self.convf2(torch.cat([x_s, x_1], dim=1)))).view(b, c, -1) # [b, c, h, w] -> [b, c, h*w]

        v_1 = self.v1(x_1).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]
        v_2 = self.v2(x_2).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]

        att1 = torch.bmm(shared_class_center1, v_1) * self.scale # [b, c, hw] x [b, hw, c] -> [b, c, c]
        att2 = torch.bmm(shared_class_center2, v_2) * self.scale  # [b, c, hw] x [b, hw, c] -> [b, c, c]

        out_1 = torch.bmm(torch.softmax(att1, dim=-1), v_1.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw]
        out_2 = torch.bmm(torch.softmax(att2, dim=-1), v_2.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw]

        x_s_ = self.unclustering(torch.cat([shared_class_center1.view(b, c, h, w), shared_class_center2.view(b, c, h, w)], dim=1)) + x_s

        return out_1 + x_2_+ out_2 + x_1_+ x_s_

class BIE_2(nn.Module):
    def __init__(self, nf=64):
        super(BIE_2, self).__init__()
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

        # initialization
        initialize_weights([self.convf1, self.convf2, self.clustering, self.unclustering, self.v1, self.v2], 0.1)

    def forward(self, x_1, x_2 ):
        # x_1, x_2 =x
        b, c, h, w = x_1.shape
        x_s = x_1+x_2
        x_1_ = self.conv1(x_1)
        x_2_ = self.conv2(x_2)
        shared_class_center1 = self.clustering(self.norm_s(self.convf1(torch.cat([x_s, x_2], dim=1)))).view(b, c, -1) # [b, c, h, w] -> [b, c, h*w]
        shared_class_center2 = self.clustering(self.norm_s(self.convf2(torch.cat([x_s, x_1], dim=1)))).view(b, c, -1) # [b, c, h, w] -> [b, c, h*w]

        v_1 = self.v1(x_1).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]
        v_2 = self.v2(x_2).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]

        att1 = torch.bmm(shared_class_center1, v_1) * self.scale # [b, c, hw] x [b, hw, c] -> [b, c, c]
        att2 = torch.bmm(shared_class_center2, v_2) * self.scale  # [b, c, hw] x [b, hw, c] -> [b, c, c]

        out_1 = torch.bmm(torch.softmax(att1, dim=-1), v_1.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw]
        out_2 = torch.bmm(torch.softmax(att2, dim=-1), v_2.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw]

        x_s_ = self.unclustering(torch.cat([shared_class_center1.view(b, c, h, w), shared_class_center2.view(b, c, h, w)], dim=1)) + x_s

        return out_1 + x_2_+ out_2 + x_1_+ x_s_

class BIEF(nn.Module):
    def __init__(self, nf=64):
        super(BIEF, self).__init__()
        # self-process
        self.conv1 = ResidualBlock_noBN(nf)
        self.conv2 = self.conv1
        self.convf1 = nn.Conv2d(nf * 2, nf, 1, 1)
        self.convf2 = self.convf1
        self.convf3 = nn.Conv2d(nf * 3, nf, 1, 1)
        self.scale = nf ** -0.5
        self.norm_s = LayerNorm2d(nf)
        self.clustering = nn.Conv2d(nf, nf, 1, 1, padding=0)
        self.unclustering = nn.Conv2d(nf * 2, nf, 1, stride=1, padding=0)

        self.v1 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)
        self.v2 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)

        # initialization
        initialize_weights([self.convf1, self.convf2, self.clustering, self.unclustering, self.v1, self.v2], 0.1)

    def forward(self, x_1, x_2):
        # x_1, x_2 =x
        b, c, h, w = x_1.shape
        x_s = x_1+x_2
        x_1_ = self.conv1(x_1)
        x_2_ = self.conv2(x_2)
        shared_class_center1 = self.clustering(self.norm_s(self.convf1(torch.cat([x_s, x_2], dim=1)))).view(b, c, -1) # [b, c, h, w] -> [b, c, h*w]
        shared_class_center2 = self.clustering(self.norm_s(self.convf2(torch.cat([x_s, x_1], dim=1)))).view(b, c, -1) # [b, c, h, w] -> [b, c, h*w]

        v_1 = self.v1(x_1).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]
        v_2 = self.v2(x_2).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]

        att1 = torch.bmm(shared_class_center1, v_1) * self.scale # [b, c, hw] x [b, hw, c] -> [b, c, c]
        att2 = torch.bmm(shared_class_center2, v_2) * self.scale  # [b, c, hw] x [b, hw, c] -> [b, c, c]

        out_1 = torch.bmm(torch.softmax(att1, dim=-1), v_1.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw]
        out_2 = torch.bmm(torch.softmax(att2, dim=-1), v_2.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw]

        x_s_ = self.unclustering(torch.cat([shared_class_center1.view(b, c, h, w), shared_class_center2.view(b, c, h, w)], dim=1)) + x_s
        out = torch.cat([out_1 + x_2_,out_2 + x_1_,x_s_],dim=1)
        out = self.convf3(out)
        return out
#输入B C H W  输出B C H W
if __name__ == "__main__":
    input1 = torch.randn(1, 30, 128, 128)
    input2 = torch.randn(1, 30, 128, 128)
    input3 = torch.randn(1, 30, 128, 128)

    model = BIE_3(30)
    output = model(input1,input2,input3)
    print('BIE_3——input_size:', input1.size())
    print('BIE_3——output_size:', output.size())

    model = BIE_2(30)
    output = model(input1,input2)
    print('BIE_2——input_size:', input1.size())
    print('BIE_2——output_size:', output.size())

    model = BIEF(30)
    output = model(input1, input2)
    print('BIEF——input_size:', input1.size())
    print('BIEF——output_size:', output.size())
