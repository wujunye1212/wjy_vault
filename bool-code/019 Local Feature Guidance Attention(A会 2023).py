import torch.nn as nn
import torch
"""
论文地址：https://arxiv.org/pdf/2309.11131
论文题目：Locate and Verify: A Two-Stream Network for Improved Deepfake Detection (CCF A会)
中文题目：用于改进Deep fake检测的双流网络
讲解视频：https://www.bilibili.com/video/BV1eCftYWENa/
    局部伪造引导注意力（Local Forgery Guided Attention, LFGA）
        现有方法严重依赖非生成图像区域进行预测的问题，因此通过使训练模型更有信心地识别任何被生成的区域，来提高分类的准确性。 
          LFGA模块从定位分支获取注意力图，引导分类分支学习更稳健和信息丰富的分类特征。
"""
class LFGA(nn.Module):  # Local Feature Guidance Attention，旨在引导特征图的注意力以更好地聚焦在局部特征上
    def __init__(self, in_channel=3, out_channel=None, ratio=4):
        super(LFGA, self).__init__()  # 调用父类nn.Module的构造函数
        self.chanel_in = in_channel  # 输入通道数

        # 如果没有指定输出通道数，则使用输入通道数除以ratio作为输出通道数，但至少为1
        if out_channel is None:
            out_channel = in_channel // ratio if in_channel // ratio > 0 else 1

        # 定义查询、键和值的卷积层
        self.query_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)  # 查询卷积
        self.key_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)  # 键卷积
        self.value_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)  # 值卷积
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习参数γ，用于调节注意力机制的影响

        self.softmax = nn.Softmax(dim=-1)  # Softmax激活函数，用于归一化能量矩阵
        self.relu = nn.ReLU()  # ReLU激活函数
        self.bn = nn.BatchNorm2d(self.chanel_in)  # 二维批归一化层

    def forward(self, fa, fb):
        B, C, H, W = fa.size()  # 获取fa的尺寸信息

        # ====================== fb =========================
        # 对fb进行查询变换，并调整维度  # CV 4  - 时序 3进行计算？  那么我们也可以通过时序3 转化 CV 4这种四维 进行计算，方便我们创新
        proj_query = self.query_conv(fb).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C
        # print("proj_query:",proj_query.shape)   # proj_query: torch.Size([16, 1024, 1])

        # 对fb进行键变换，并调整维度
        proj_key = self.key_conv(fb).view(B, -1, H * W)  # B, C, HW
        # print("proj_key:", proj_key.shape)  # proj_key: torch.Size([16, 1, 1024])

        # 计算能量矩阵（点积）
        energy = torch.bmm(proj_query, proj_key)  # B, HW, HW
        # print("energy:", energy.shape)  # energy: torch.Size([16, 1024, 1024])

        # 应用Softmax得到注意力权重
        attention = self.softmax(energy)  # B, HW, HW
        # print("attention:", attention.shape)    # attention: torch.Size([16, 1024, 1024])
        # ===================================================

        # ====================== fa =========================
        # 对fa进行值变换，并调整维度
        proj_value = self.value_conv(fa).view(B, -1, H * W)  # B, C, HW
        # print("proj_value:", proj_value.shape) # torch.rand(16, 3, 32, 32) ===》proj_value: torch.Size([16, 3, 1024])
        # ===================================================

        # 注意力加权求和
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B, C, HW
        # print("out:", out.shape)  #  out: torch.Size([16, 3, 1024])
        out = out.view(B, C, H, W)  # 恢复原始空间维度 16, 3, 32, 32)

        # 将加权后的输出与原始输入fa结合，并通过可学习参数γ调节
        out = self.gamma * out + fa

        return self.relu(out)  # 返回经过ReLU激活后的结果

if __name__ == '__main__':
    block = LFGA(in_channel=3, ratio=4)  # 实例化LFGA模块
    fa = torch.rand(16, 3, 32, 32)  # 创建随机张量fa作为输入
    fb = torch.rand(16, 3, 32, 32)  # 创建随机张量fb作为输入

    output = block(fa, fb)  # 将fa和fb通过LFGA模块进行前向传播
    print(fa.size())  # 输出原始fa的大小
    print(fb.size())  # 输出原始fb的大小
    print(output.size())  # 输出处理后output的大小

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")