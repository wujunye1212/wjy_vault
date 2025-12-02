import torch.nn as nn
import torch
import torch.nn.functional as F
'''
论文地址：https://arxiv.org/pdf/2309.11131
论文题目：Locate and Verify: A Two-Stream Network for Improved Deepfake Detection (CCF A会)
中文题目：用于改进Deep fake检测的双流网络
讲解视频：https://www.bilibili.com/video/BV1iKw8eMEnY/
    跨模态一致性增强（Cross - modality Consistency Enhancement, CMCE）
          CMCE 模块从 RGB 和 SRM 模态中学习组合特征表示，确保两个分支尽可能保留各自的特征，同时捕捉两种模态之间的相互作用和相互影响。
          杜绝直接连接或注意力加权增强来合并两种模态。
    优点：
        （1）与单模态相比，CMCE 模块学习到了更丰富的伪造特征。
        （2）它还为两种模态保留了独立且具有代表性的特征。
'''

class CMCE(nn.Module):  # Contrastive Multimodal Contrastive Enhancement，用于增强模型对特征的关注度，提高模型的性能
    def __init__(self):
        super(CMCE, self).__init__()  # 调用父类nn.Module的构造函数
        self.relu = nn.ReLU()  # 定义ReLU激活函数

    def forward(self, fa, fb):
        (b1, c1, h1, w1), (b2, c2, h2, w2) = fa.size(), fb.size()  # 获取输入fa和fb的尺寸信息
        assert c1 == c2  # 确保fa和fb在通道数上一致

        cos_sim = F.cosine_similarity(fa, fb, dim=1)  # 计算fa和fb在通道维度上的余弦相似性
        cos_sim = cos_sim.unsqueeze(1)  # 增加一个维度以匹配后续操作

        # 使用余弦相似性调整特征图fa和fb
        fa = fa + fb * cos_sim
        fb = fb + fa * cos_sim

        # 对调整后的特征图应用ReLU激活
        fa = self.relu(fa)
        fb = self.relu(fb)

        # f = fa + fb
        return fa, fb  # 返回经过处理的fa和fb

if __name__ == '__main__':
    block = CMCE()  # 实例化CMCE模块
    fa = torch.rand(16, 3, 32, 32)  # 创建随机张量fa作为输入
    fb = torch.rand(16, 3, 32, 32)  # 创建随机张量fb作为输入

    fa1, fb1 = block(fa, fb)  # 将fa和fb通过CMCE模块进行前向传播
    print(fa.size())  # 输出原始fa的大小
    print(fb.size())  # 输出原始fb的大小

    print(fa1.size())  # 输出处理后fa1的大小
    print(fb1.size())  # 输出处理后fb1的大小

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")