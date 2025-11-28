import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
# pip install pytorch_wavelets==1.3.0
# pip install PyWavelets

"""
论文地址：https://www.sciencedirect.com/science/article/pii/S0031320323005174
论文题目：Haar wavelet downsampling: A simple but effective downsampling module for semantic segmentation (1区Top)
中文题目：Haar小波下采样：一个简单但有效的语义分割下采样模块
讲解视频：https://www.bilibili.com/video/BV1Cv2WYFEnH/
    ①难点：1、在卷积神经网络中，下采样操作如最大池化和卷积有助于特征聚合、扩展感受野及降低计算量。但在语义分割任务中，这些操作可能导致重要空间信息的丢失，
                                    影响边界、尺度和纹理等关键细节的保持。
          2、虽然一些技术如跳过连接、多尺度特征融合和先验知识应用试图缓解这一问题，但恢复由传统下采样方法引起的信息损失依然是一个挑战。
          3、简单的特征聚合方式可能引入无关信息，从而妨碍网络学习具有区分性的特征。
    ②主要动机：通过使用Haar小波变换增加特征图的通道数并减少其分辨率而不损失信息（无损下采样），然后采用卷积操作进行代表特征学习以过滤冗余信息。
    ③未来改进：由于卷积操作的局部性质，HWD模块缺乏捕获全局上下文并建立长距离空间关系的能力（CNN + Transformer）。         
"""
class HWD(nn.Module):
    # 初始化函数
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        # 定义离散小波变换(DWT)前向操作，参数J表示分解级别为1，mode设置边界处理方式为零填充，wave指定使用Haar小波
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        # 定义卷积-批归一化-激活层序列
        self.conv_bn_relu = nn.Sequential(
            # 添加2D卷积层，输入通道数是原通道数的4倍（因为经过DWT后会产生4个子带），输出通道数为out_ch
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            # 批量归一化层，对out_ch个特征图进行归一化
            nn.BatchNorm2d(out_ch),
            # ReLU激活函数，inplace=True表示直接在输入数据上进行修改以节省内存
            nn.ReLU(inplace=True),
        )
    # 前向传播函数
    def forward(self, x):
        # 对输入x执行DWT，得到低频分量yL和高频分量yH
        yL, yH = self.wt(x)     # torch.Size([1, 3, 32, 32]) torch.Size([1, 3, 32, 32])

        # 从高频分量yH中提取出水平细节系数
        y_HL = yH[0][:, :, 0, ::]   # torch.Size([1, 3, 32, 32])
        # 从高频分量yH中提取出垂直细节系数
        y_LH = yH[0][:, :, 1, ::]   # torch.Size([1, 3, 32, 32])
        # 从高频分量yH中提取出对角线细节系数
        y_HH = yH[0][:, :, 2, ::]   # torch.Size([1, 3, 32, 32])

        # 【改进点】 是否可以改为低频？
        # 相关视频：https://www.bilibili.com/video/BV1pJsCejEzm/
        #         https://www.bilibili.com/video/BV1T5seefEAo/

        # 将低频分量yL与三个方向的高频分量沿着通道维度拼接
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)    # torch.Size([1, 12, 32, 32])

        # 拼接后的张量通过定义好的conv_bn_relu序列
        x = self.conv_bn_relu(x)
        # 返回最终输出
        return x

# 主程序入口
if __name__ == '__main__':
    # 创建HWD实例，设定输入通道数为3，输出通道数也为3
    block = HWD(in_ch=3, out_ch=3)
    # 生成随机输入张量，形状为(1, 3, 64, 64)
    input = torch.rand(1, 3, 64, 64)
    # 通过HWD模块处理输入数据，获取输出
    output = block(input)
    # 打印输入张量的尺寸
    print('input :', input.size())
    # 打印输出张量的尺寸
    print('output :', output.size())