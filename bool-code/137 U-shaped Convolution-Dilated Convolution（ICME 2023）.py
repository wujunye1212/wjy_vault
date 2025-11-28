import torch
import torch.nn as nn
"""
    论文地址：https://arxiv.org/pdf/2303.10321
    论文题目：ABC: Attention with Bilinear Correlation for Infrared Small Target Detection（CCF B）
    中文题目：ABC：用于红外小目标检测的双线性相关注意力机制（CCF B）
    讲解视频：https://www.bilibili.com/video/BV11xoQYKExV/
    U形扩张卷积（U-shaped Convolution-Dilated Convolution, UCDC）：
        实际意义：①深层特征语义信息少：深层特征包含的语义信息相对较少，在进行卷积操作时容易造成信息丢失。
                ②分辨率低与感受野：特征图分辨率变小，卷积操作的感受野相对较大，常规的卷积操作难以精细处理这些深层特征，无法充分挖掘其中的有用信息。
        实现方式：以代码为准。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】   
"""

def conv_relu_bn(in_channel, out_channel, dirate):
    return nn.Sequential(
        # 定义一个二维卷积层
        # in_channels：输入通道数
        # out_channels：输出通道数
        # kernel_size：卷积核大小为3x3
        # stride：步长为1
        # padding：填充大小等于膨胀率，保证特征图尺寸不变
        # dilation：膨胀率
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=dirate,
                  dilation=dirate),
        # 批量归一化层，用于加速模型收敛和提高稳定性
        nn.BatchNorm2d(out_channel),
        # ReLU激活函数，inplace=True表示直接在输入上修改，节省内存
        nn.ReLU(inplace=True)
    )

class UCDC(nn.Module):
    def __init__(self, in_ch, out_ch):
        # 调用父类的构造函数
        super(UCDC, self).__init__()
        # 第一个普通卷积层，输入通道数为in_ch，输出通道数为out_ch，膨胀率为1
        self.conv1 = conv_relu_bn(in_ch, out_ch, 1)
        # 第一个膨胀卷积层，输入通道数为out_ch，输出通道数为out_ch的一半，膨胀率为2
        self.dconv1 = conv_relu_bn(out_ch, out_ch // 2, 2)
        # 第二个膨胀卷积层，输入通道数为out_ch的一半，输出通道数为out_ch的一半，膨胀率为4
        self.dconv2 = conv_relu_bn(out_ch // 2, out_ch // 2, 4)
        # 第三个膨胀卷积层，输入通道数为out_ch，输出通道数为out_ch，膨胀率为2
        self.dconv3 = conv_relu_bn(out_ch, out_ch, 2)
        # 第二个普通卷积层，输入通道数为out_ch的两倍，输出通道数为out_ch，膨胀率为1
        self.conv2 = conv_relu_bn(out_ch * 2, out_ch, 1)

    def forward(self, x):
        # 通过第一个普通卷积层得到特征图x1
        x1 = self.conv1(x)
        # 将x1输入到第一个膨胀卷积层得到特征图dx1
        dx1 = self.dconv1(x1)
        # 将dx1输入到第二个膨胀卷积层得到特征图dx2
        dx2 = self.dconv2(dx1)
        # 将dx1和dx2在通道维度上拼接，然后输入到第三个膨胀卷积层得到特征图dx3
        dx3 = self.dconv3(torch.cat((dx1, dx2), dim=1))
        # 将x1和dx3在通道维度上拼接，然后输入到第二个普通卷积层得到最终输出
        out = self.conv2(torch.cat((x1, dx3), dim=1))
        return out

if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = UCDC(64, 64)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")