import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms.functional import rgb_to_grayscale

'''
MEGANet：用于弱边界息肉分割的多尺度边缘引导注意力网络 （WACV 2024）
视频内容：简单介绍一下该模块+教大家缝合该MEGA模块过程中避免程序报错

医疗保健中的高效息肉分割在实现结直肠癌的早期诊断方面起着关键作用。
然而，息肉的分割带来了许多挑战， 包括背景的复杂分布、息肉大小和形状的变化以及模糊的边界。
定义前景（即息肉本身）和背景（周围组织）之间的边界是困难的。
为了缓解这些挑战，我们提出了专为结肠镜检查图像中的息肉分割量身定制的 M终极规模 Edge-Guided Attention Network （MEGANet）。
该网络从经典边缘检测技术与注意力机制的融合中汲取灵感。

通过结合这些技术，MEGANet 有效地保留了高频信息，尤其是边缘和边界，这些信息往往会随着神经网络的深入而受到侵蚀。

对五个基准数据集进行的广泛定性和定量实验表明，我们的MEGANet在六个评估指标下优于其他现有的 SOTA 方法。

'''
def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel
def downsample(x):
    return x[:, :, ::2, ::2]
def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out
def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))
def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff
def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

# Multi Edge-Guided Attention Module
class MEGA(nn.Module):
    def __init__(self, in_channels):
        super(MEGA, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.cbam = CBAM(in_channels)

    # def forward(self, edge_feature, x, pred):
    def forward(self, org_img, x, pred):
        grayscale_img = rgb_to_grayscale(org_img)
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        edge_feature = edge_feature[1]

        residual = x
        xsize = x.size()[2:]
        pred = torch.sigmoid(pred)

        # reverse attention
        background_att = 1 - pred
        background_x = x * background_att

        # boudary attention
        edge_pred = make_laplace(pred, 1)
        pred_feature = x * edge_pred

        # high-frequency feature
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)

        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        out = self.cbam(out)
        return out
# 输入 B C H W,  输出B C H W
if __name__ == "__main__":
    # 定义原始输入图像的尺寸 (batch_size, channels, height, width)
    batch_size = 1
    in_channels = 3
    height, width = 256, 256

    # 随机生成输入org_img图像、 x特征图和pred预测掩码
    org_img = torch.randn(batch_size, in_channels, height, width).cuda()
    x = torch.randn(batch_size, 16, height // 4, width // 4).cuda()
    pred = torch.randn(batch_size, 1, height // 4, width // 4).cuda()

    # 创建 MEGA 模块的实例
    mega_module = MEGA(in_channels=16).cuda()  #对x特征图通道数要求大于16，为什么？

    # 使用 MEGA 模块进行前向传播
    output = mega_module(org_img, x, pred)

    # 打印输入和输出的张量
    print('Input size:', x.size())
    print('Output size:', output.size())
