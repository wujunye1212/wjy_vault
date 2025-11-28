import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
    论文地址：https://arxiv.org/pdf/2406.03751
    论文题目：Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting（AAAI 2025）
    中文题目：用于时间序列预测的自适应多尺度分解框架(AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV1SVc9eoEbG/
    双依赖交互块（Dual Dependency Interaction Block, DDI）
         发现问题：单个尺度信息建模会忽略多尺度关系。在现实中，不同尺度是相互作用的，例如在股票价格序列中，月度经济趋势会影响每日的股价波动，而这些月度趋势又受到年度市场周期的影响。
         解决思路：时间依赖关系对不同时间段间相互作用进行建模，而跨通道依赖关系对不同变量间关系进行建模，从而增强时间序列表征能力。
"""
class DDI(nn.Module):
    def __init__(self, input_shape, dropout=0.2, patch=12, alpha=0.0, layernorm=True):
        super(DDI, self).__init__()
        # 输入形状，第一个维度为序列长度，第二个维度为特征数量
        self.input_shape = input_shape
        if alpha > 0.0:
            # 计算前馈层的维度
            self.ff_dim = 2 ** math.ceil(math.log2(self.input_shape[-1]))
            self.fc_block = nn.Sequential(
                nn.Linear(self.input_shape[-1], self.ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_dim, self.input_shape[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.n_history = 1  # 历史步数
        self.alpha = alpha  # 控制残差连接的系数
        self.patch = patch  # 每个补丁的长度

        self.layernorm = layernorm  # 是否使用层归一化
        if self.layernorm:
            self.norm = nn.BatchNorm1d(self.input_shape[0] * self.input_shape[-1])
        self.norm1 = nn.BatchNorm1d(self.n_history * patch * self.input_shape[-1])
        if self.alpha > 0.0:
            self.norm2 = nn.BatchNorm1d(self.patch * self.input_shape[-1])

        self.agg = nn.Linear(self.n_history * self.patch, self.patch)  # 聚合层
        self.dropout_t = nn.Dropout(dropout)  # Dropout层

    def forward(self, x):
        # 输入形状：[batch_size, feature_num, seq_len]
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)

        output = torch.zeros_like(x)  # 初始化输出
        output[:, :, :self.n_history * self.patch] = x[:, :, :self.n_history * self.patch].clone()

        for i in range(self.n_history * self.patch, self.input_shape[0], self.patch):
            # 提取输入块
            input = output[:, :, i - self.n_history * self.patch: i]
            # 归一化输入块
            input = self.norm1(torch.flatten(input, 1, -1)).reshape(input.shape)
            # 聚合操作
            input = F.gelu(self.agg(input))  # 将历史块聚合为当前块
            input = self.dropout_t(input)  # 应用Dropout

            # 计算残差
            tmp = input + x[:, :, i: i + self.patch]
            res = tmp  # 保存中间结果

            # 如果alpha大于0，则应用额外的前馈网络
            if self.alpha > 0.0:
                tmp = self.norm2(torch.flatten(tmp, 1, -1)).reshape(tmp.shape)
                tmp = torch.transpose(tmp, 1, 2)  # 转置
                tmp = self.fc_block(tmp)  # 前馈网络
                tmp = torch.transpose(tmp, 1, 2)  # 转置回去

            output[:, :, i: i + self.patch] = res + self.alpha * tmp  # 更新输出

        return output

if __name__ == "__main__":
    # 定义输入形状和其他参数
    input_shape = (48, 10)  # 示例：序列长度=48，特征数量=10
    dropout = 0.2
    patch = 12
    alpha = 0.1
    layernorm = True

    # 初始化DDI模型
    model = DDI(input_shape=input_shape, dropout=dropout, patch=patch, alpha=alpha, layernorm=layernorm)
    batch_size = 16
    x = torch.randn(batch_size, input_shape[1], input_shape[0])  # 输入形状：[batch_size, feature_num, seq_len]
    print("Input shape:", x.shape)
    # 通过模型进行前向传递
    output = model(x)
    # 打印输出形状
    print("Output shape:", output.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")