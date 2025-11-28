import torch
from torch import nn

"""
    论文地址：https://arxiv.org/pdf/1904.02998v1.pdf
    论文题目：Relation-Aware Global Attention for Person Re-identiﬁcation (CVPR)
    中文题目：用于行人重识别的关系感知全局注意力
    讲解视频：https://www.bilibili.com/video/BV1tMCBYPEKB/
    关系感知全局注意力模块（Relation-Aware Global Attention Module,  RAGAM）
         空间关系感知全局注意力：每个空间位置特征向量视为节点，计算节点间亲和力矩阵，以光栅扫描顺序堆叠关系得到关系向量，与特征向量嵌入形成空间关系感知特征，进而计算空间注意力值。
         通道关系感知全局注意力：每个通道特征图作为节点，计算节点间亲和力矩阵，以光栅扫描顺序堆叠关系得到关系向量，与特征向量得到通道关系感知特征，进而计算通道注意力值。

"""
class RGA(nn.Module):
    def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True,
                 cha_ratio=8, spa_ratio=8, down_ratio=8):
        super(RGA, self).__init__()

        self.in_channel = in_channel  # 输入通道数
        self.in_spatial = in_spatial  # 输入空间大小（H*W）

        self.use_spatial = use_spatial  # 是否使用空间注意力
        self.use_channel = use_channel  # 是否使用通道注意力

        self.inter_channel = max(in_channel // cha_ratio, 1)  # 中间通道数
        self.inter_spatial = max(in_spatial // spa_ratio, 1)  # 中间空间大小

        # 原始特征的嵌入函数
        if self.use_spatial:
            self.gx_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.gx_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

        # 关系特征的嵌入函数
        if self.use_spatial:
            self.gg_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
        if self.use_channel:
            self.gg_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )

        # 学习注意力权重的网络
        if self.use_spatial:
            num_channel_s = 1 + self.inter_spatial
            self.W_spatial = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_s // down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_s // down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )
        if self.use_channel:
            num_channel_c = max(1 + self.inter_channel, 1)  # 确保至少为1
            self.W_channel = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_c, out_channels=max(num_channel_c // down_ratio, 1),
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(max(num_channel_c // down_ratio, 1)),
                nn.ReLU(),
                nn.Conv2d(in_channels=max(num_channel_c // down_ratio, 1), out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )

        # 用于建模关系的嵌入函数
        if self.use_spatial:
            self.theta_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            self.phi_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.theta_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            self.phi_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

    def forward(self, x):
        b, c, h, w = x.size()

        if self.use_spatial:
            # 空间注意力
            # Q
            theta_xs = self.theta_spatial(x)
            theta_xs = theta_xs.view(b, self.inter_channel, -1)
            theta_xs = theta_xs.permute(0, 2, 1)
            # K
            phi_xs = self.phi_spatial(x)
            phi_xs = phi_xs.view(b, self.inter_channel, -1)
            Gs = torch.matmul(theta_xs, phi_xs)

            # 以光栅扫描顺序堆叠关系得到关系向量
            # 第一部分 cat
            Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)
            Gs_out = Gs.view(b, h * w, h, w)
            Gs_joint = torch.cat((Gs_in, Gs_out), 1)
            Gs_joint = self.gg_spatial(Gs_joint)
            # 第二部分 cat
            g_xs = self.gx_spatial(x)
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)
            ys = torch.cat((g_xs, Gs_joint), 1)

            W_ys = self.W_spatial(ys)

            if not self.use_channel:
                out = torch.sigmoid(W_ys.expand_as(x)) * x
                return out
            else:
                x = torch.sigmoid(W_ys.expand_as(x)) * x

        if self.use_channel:
            # 通道注意力
            xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
            # Q
            theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
            # K
            phi_xc = self.phi_channel(xc).squeeze(-1)
            Gc = torch.matmul(theta_xc, phi_xc)

            # 以光栅扫描顺序堆叠关系得到关系向量
            # 第一部分 cat
            Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
            Gc_out = Gc.unsqueeze(-1)
            Gc_joint = torch.cat((Gc_in, Gc_out), 1)
            Gc_joint = self.gg_channel(Gc_joint)
            # 第二部分 cat
            g_xc = self.gx_channel(xc)
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)
            yc = torch.cat((g_xc, Gc_joint), 1)

            W_yc = self.W_channel(yc).transpose(1, 2)
            out = torch.sigmoid(W_yc) * x
            return out

if __name__ == '__main__':
    block = RGA(in_channel=3, in_spatial=64*64)  # 注意这里in_spatial应当是H*W
    input = torch.rand(32, 3, 64, 64)
    output = block(input)
    print("Input size:", input.size())
    print("Output size:", output.size())

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
