import random
import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2403.10067
    论文题目：MixStyle Neural Networks for Domain Generalization and Adaptation (CCF-A)
    中文题目：用于领域泛化与领域适应的 MixStyle 神经网络 (CCF-A)
    讲解视频：https://www.bilibili.com/video/BV1BNWyzcEGw/
    MixStyle神经网络模块/特征泛化增强模块（MixStyle Neural Networks）：
        实际意义：①跨域泛化能力不足：传统卷积神经网络（CNN）在训练和测试数据服从相同分布的情况下表现优异，但在遇到分布不同的目标域时性能会急剧下降。
                ②低标注数据下的泛化能力不足：无标注数据的 “伪标签噪声”易导致模型过拟合，无法有效提升分布不同时的泛化能力。
        实现方式：MixStyle 通过混合多张图像的特征统计信息的均值、标准差，在特征空间中生成“虚拟风格”，从而提升模型跨域泛化能力。
"""

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        super().__init__()
        # p：表示执行MixStyle的概率（例如0.5表示50%的概率启用）
        self.p = p
        # Beta分布，用于生成混合比例lambda，控制两种风格的融合程度
        self.beta = torch.distributions.Beta(alpha, alpha)
        # eps：用于数值稳定，防止除以0
        self.eps = eps
        # alpha：Beta分布的形状参数，控制lambda的分布形态
        self.alpha = alpha
        # mix：混合策略，可选'random'或'crossdomain'
        self.mix = mix
        # 是否启用MixStyle模块（可通过函数手动开启或关闭）
        self._activated = True

    def forward(self, x):
        # 前向传播定义输入输出的计算逻辑
        # x形状：[B, C, H, W]
        if not self.training or not self._activated:
            # 如果模型不是训练模式或未激活，则直接返回原输入
            return x

        # 以概率p决定是否进行MixStyle混合
        if random.random() > self.p:
            return x

        B = x.size(0)  # 批次大小

        # 计算每个样本的通道均值和方差
        mu = x.mean(dim=[2, 3], keepdim=True)  # 在空间维度(H, W)上求均值
        var = x.var(dim=[2, 3], keepdim=True)  # 在空间维度(H, W)上求方差
        sig = (var + self.eps).sqrt()          # 计算标准差，并加上eps防止数值不稳定
        # detach：截断梯度，防止这些统计量参与反向传播
        mu, sig = mu.detach(), sig.detach()

        # 对输入进行标准化，使其均值为0，方差为1
        x_normed = (x - mu) / sig

        # 从Beta分布中采样lambda（混合系数）
        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)  # 确保lambda与输入张量在同一设备上（如GPU）

        # 根据选择的混合方式确定样本配对顺序
        if self.mix == 'random':
            # 随机打乱批次中的样本顺序，实现随机混合
            perm = torch.randperm(B)
        elif self.mix == 'crossdomain':
            # 跨域混合：将批次分为两半并对调顺序
            perm = torch.arange(B - 1, -1, -1)  # 反向索引，例如 [7,6,5,4,3,2,1,0]
            perm_b, perm_a = perm.chunk(2)      # 拆分成两半
            perm_b = perm_b[torch.randperm(B // 2)]  # 打乱后一半
            perm_a = perm_a[torch.randperm(B // 2)]  # 打乱前一半
            perm = torch.cat([perm_b, perm_a], 0)    # 拼接为新的排列
        else:
            # 未实现的模式则报错
            raise NotImplementedError

        # 取出打乱顺序后的均值和方差（即另一个样本的风格特征）
        mu2, sig2 = mu[perm], sig[perm]

        # 按比例lambda混合两个样本的风格统计特征
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        # 用混合后的风格重新调整归一化特征，得到输出
        return x_normed * sig_mix + mu_mix

if __name__ == '__main__':
    # p：表示执行MixStyle的概率（例如0.5表示50%的概率启用）
    # Beta分布，用于生成混合比例lambda，控制两种风格的融合程度
    # alpha：Beta分布的形状参数，控制lambda的分布形态
    model = MixStyle(p=0.5, alpha=0.1, eps=1e-6, mix='random')
    input = torch.rand(2, 32, 50, 50)
    output = model(input)
    print(f"输入张量形状: {input.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")