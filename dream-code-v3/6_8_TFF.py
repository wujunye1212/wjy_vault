import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )



class TFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TFF, self).__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xA, xB):
        x_diff = xA - xB  # 通过相减获得粗略的变化表示: (B,C,H,W)

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1)) #将变化特征与xA拼接,通过DWConv提取特征: (B,C,H,W)-cat-(B,C,H,W)-->(B,2C,H,W);  (B,2C,H,W)-catconvA-->(B,C,H,W)
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1)) #将变化特征与xB拼接,通过DWConv提取特征: (B,C,H,W)-cat-(B,C,H,W)-->(B,2C,H,W);  (B,2C,H,W)-catconvB-->(B,C,H,W)

        A_weight = self.sigmoid(self.convA(x_diffA)) # 通过卷积映射到1个通道,生成空间描述符,然后通过sigmoid生成权重: (B,C,H,W)-convA->(B,1,H,W)
        B_weight = self.sigmoid(self.convB(x_diffB)) # 通过卷积映射到1个通道,生成空间描述符,然后通过sigmoid生成权重: (B,C,H,W)-convB->(B,1,H,W)

        xA = A_weight * xA # 使用生成的权重A_weight调整对应输入xA: (B,1,H,W) * (B,C,H,W) == (B,C,H,W)
        xB = B_weight * xB # 使用生成的权重B_weight调整对应输入xB: (B,1,H,W) * (B,C,H,W) == (B,C,H,W)

        x = self.catconv(torch.cat([xA, xB], dim=1)) # 两个特征拼接,然后恢复与输入相同的shape: (B,C,H,W)-cat-(B,C,H,W)-->(B,2C,H,W); (B,2C,H,W)--catconv->(B,C,H,W)

        return x


if __name__ == '__main__':
    # (B,C,H,W)
    x1 = torch.randn(1,64,224,224).to(device)
    x2 = torch.randn(1,64,224,224).to(device)

    Model = TFF(in_channel=64, out_channel=64).to(device)
    out = Model(x1,x2)
    print(out.shape)