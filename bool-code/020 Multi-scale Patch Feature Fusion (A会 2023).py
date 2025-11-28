import torch.nn as nn
import torch

"""
论文地址：https://arxiv.org/pdf/2309.11131
论文题目：Locate and Verify: A Two-Stream Network for Improved Deepfake Detection (CCF A会)
中文题目：用于改进Deep fake检测的双流网络
讲解视频：https://www.bilibili.com/video/BV1rCfiYxEGv/
    多尺度补丁特征融合（Multi-scale Patch Feature Fusion，MPFF）
        在许多现有的深度伪造检测工作中，伪造方法所导致的伪影（即与真实图像不一致的特征或痕迹）在图像的浅层特征（轮廓形状）更为显著，
        应该更加关注浅层特征中更明显的伪影，以提高深度伪造检测的性能。
        
    1、分类特征关注全局语义信息：在深度伪造检测中，分类特征的目的是确定图像是否被伪造，因此需要理解图像的全局语义信息，例如物体的类别、场景的背景等。
    2、定位特征关注局部空间细节：那些可能存在伪造痕迹的区域。例如，在人脸的眼睛、嘴巴、鼻子等，可以帮助更精确地检测这些局部区域的异常。
    3、每个图像补丁位置信息的重要性：在深度伪造检测中，图像通常会被分割成多个token。如果模型无法准确知道每个补丁在图像中的位置，就可能会导致对伪造区域的误判或漏判。
            此外，位置信息还可以帮助模型更好地融合不同尺度的特征，从而提高检测的准确性。
            
            例如，在多尺度补丁特征融合中，通过维护位置信息，可以将不同尺度下的特征准确地对应到图像的相应位置，从而实现更有效的特征融合。
            【如何维护特征位置信息】
"""
class MPFF(nn.Module):
    def __init__(self, size=8):
        super(MPFF, self).__init__()  # 调用父类nn.Module的构造函数，初始化网络模块
        self.size = size  # 定义每个块的大小

    def forward(self, fa, fb):
        b1, c1, h1, w1 = fa.size()  # 获取输入张量fa的尺寸信息：批量数、通道数、高度和宽度
        b2, c2, h2, w2 = fb.size()  # 获取输入张量fb的尺寸信息：批量数、通道数、高度和宽度
        assert b1 == b2 and c1 == c2 and self.size == h2  # 确保fa和fb在批量数、通道数上一致，并且fb的高度等于预设的size

        # ===== 使得fat的高度可以被size整除 || fa填充至fb大小 =====
        padding = abs(h1 % self.size - self.size) % self.size  # 计算需要的填充量，
        pad = nn.ReplicationPad2d(padding=(padding // 2, (padding + 1) // 2, padding // 2, (padding + 1) // 2)).to(fa.device)  # 创建复制填充层，确保填充均匀分布在四周
        fa = pad(fa)  # 对fa进行填充操作，使其高度能被size整除
        b1, c1, h1, w1 = fa.size()  # 重新获取填充后的fa尺寸信息    # torch.Size([16, 3, 64, 64])
        assert h1 % self.size == 0  # 再次检查填充后fa的高度确实可以被size整除 fa == fb ????

        window = h1 // self.size  # 根据新的fa高度计算每个块的窗口大小
        fb = fb.repeat_interleave(window, dim=2)  # 在高度维度上重复fb的内容，使它与处理后的fa相匹配  torch.Size([16, 3, 64, 64])
        fb = fb.repeat_interleave(window, dim=3)  # 在宽度维度上同样地重复fb内容         torch.Size([16, 3, 64, 64])

        ff = torch.tanh(fa * fb)  # 将fa与扩展后的fb逐元素相乘后，通过tanh激活函数，得到ff    # torch.Size([16, 3, 64, 64])
        ff = torch.sum(ff, dim=1, keepdim=True)  # 沿着通道维度求和，并保持维度不变（保留一个通道） # torch.Size([16, 1, 64, 64])
        unfold = nn.Unfold(kernel_size=window, dilation=1, padding=0, stride=window)  # 创建展开操作实例，用于将图像分割成小块
        ff = unfold(ff).view(b1, -1, self.size, self.size)  # 展开ff并重塑为指定形状，其中-1表示自动计算该维度大小以适应总元素数量
        # torch.Size([16, 1, 64, 64])

        return ff  # 返回最终处理后的张量ff

if __name__ == '__main__':

    # block = MPFF(size=32)  # 实例化MPFF模块，设置每个块的大小为32
    # fa = torch.rand(16, 3, 32, 32)  # 创建随机张量fa作为输入
    # fb = torch.rand(16, 3, 32, 32)  # 创建随机张量fb作为输入

    # 跨尺度融合？
    block = MPFF(size=64)  # 实例化MPFF模块，设置每个块的大小为32
    fa = torch.rand(16, 3, 32, 32)  # 创建随机张量fa作为输入
    fb = torch.rand(16, 3, 64, 64)  # 创建随机张量fb作为输入

    output = block(fa, fb)  # 将fa和fb通过MPFF模块进行前向传播
    print(fa.size())  # 输出原始fa的大小
    print(fb.size())  # 输出原始fb的大小
    print(output.size())  # 输出处理后output的大小

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")