import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2406.03751
    论文题目：Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting（AAAI 2025）
    中文题目：用于时间序列预测的自适应多尺度分解框架(AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV1QYceeGE2U/
        自适应多预测器合成块（Adaptive Multi-predictor Synthesis Block, AMS）
             发现问题：混合专家（MoE）具有自适应特性，可以针对不同时间模式设计不同的预测器，以提高预测的准确性和泛化能力。
             解决思路：时间模式选择器（TP-Selector）对不同的时间模式进行分解并生成选择器权重S。
                    同时，时间模式投影（TP-Projection）对多个预测结果进行合成，并根据特定权重自适应地聚合输出。
"""
class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, noise_epsilon=1e-5):
        super(TopKGating, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)  # 门控线性层
        self.top_k = top_k  # 选择的专家数量
        self.noise_epsilon = noise_epsilon  # 噪声的最小值
        self.num_experts = num_experts  # 专家数量
        self.w_noise = nn.Parameter(torch.zeros(num_experts, num_experts), requires_grad=True)  # 噪声权重
        self.softplus = nn.Softplus()  # Softplus激活函数
        self.softmax = nn.Softmax(1)  # Softmax函数

    def decompostion_tp(self, x, alpha=10):
        # 输入形状：[batch_size, seq_len]
        output = torch.zeros_like(x)  # 初始化输出
        kth_largest_val, _ = torch.kthvalue(x, self.num_experts - self.top_k + 1)  # 获取第k大的值
        kth_largest_mat = kth_largest_val.unsqueeze(1).expand(-1, self.num_experts)  # 扩展为矩阵
        mask = x < kth_largest_mat  # 创建掩码
        x = self.softmax(x)  # 应用Softmax
        output[mask] = alpha * torch.log(x[mask] + 1)  # 小于k大值的处理
        output[~mask] = alpha * (torch.exp(x[~mask]) - 1)  # 大于等于k大值的处理
        return output

    def forward(self, x):
        # 输入形状：[batch_size, seq_len]
        x = self.gate(x)  # 通过线性层
        clean_logits = x  # 干净的logits

        if self.training:
            raw_noise_stddev = x @ self.w_noise  # 计算噪声标准差
            noise_stddev = ((self.softplus(raw_noise_stddev) + self.noise_epsilon))  # 应用Softplus
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)  # 加噪声
            logits = noisy_logits
        else:
            logits = clean_logits
        logits = self.decompostion_tp(logits)  # 分解
        gates = self.softmax(logits)  # 应用Softmax得到门控值
        return gates

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.2):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 输入到隐藏层
            nn.GELU(),  # GELU激活
            nn.Dropout(dropout),  # Dropout
            nn.Linear(hidden_dim, output_dim)  # 隐藏层到输出层
        )

    def forward(self, x):
        return self.net(x)  # 前向传播

class AMS(nn.Module):
    def __init__(self, input_shape, pred_len, ff_dim=2048, dropout=0.2, loss_coef=1.0, num_experts=4, top_k=2):
        super(AMS, self).__init__()
        self.num_experts = num_experts  # 专家数量
        self.top_k = top_k  # 选择的专家数量
        self.pred_len = pred_len  # 预测长度

        self.gating = TopKGating(input_shape[0], num_experts, top_k)  # 初始化门控

        self.experts = nn.ModuleList(
            [Expert(input_shape[0], pred_len, hidden_dim=ff_dim, dropout=dropout) for _ in range(num_experts)]
        )  # 初始化专家

        self.loss_coef = loss_coef  # 损失系数
        assert (self.top_k <= self.num_experts)  # 确保top_k小于或等于专家数量

    def cv_squared(self, x):
        eps = 1e-10  # 避免除零
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)  # 如果只有一个专家返回0
        return x.float().var() / (x.float().mean() ** 2 + eps)  # 计算变异系数平方

    def forward(self, x, time_embedding):
        # 输入形状：[batch_size, feature_num, seq_len]
        batch_size = x.shape[0]
        feature_num = x.shape[1]

        # 转置输入和时间嵌入
        x = torch.transpose(x, 0, 1)
        time_embedding = torch.transpose(time_embedding, 0, 1)

        output = torch.zeros(feature_num, batch_size, self.pred_len).to(x.device)  # 初始化输出
        loss = 0  # 初始化损失

        for i in range(feature_num):
            input = x[i]
            time_info = time_embedding[i]
            # 获取门控值
            gates = self.gating(time_info)

            # 初始化专家输出
            expert_outputs = torch.zeros(self.num_experts, batch_size, self.pred_len).to(x.device)

            for j in range(self.num_experts):
                expert_outputs[j, :, :] = self.experts[j](input)  # 通过专家

            expert_outputs = torch.transpose(expert_outputs, 0, 1)

            # 扩展门控值 【对不同的时间模式进行分解并生成选择器权重S】
            gates = gates.unsqueeze(-1).expand(-1, -1, self.pred_len)
            # 计算批量输出【时间模式投影（TP-Projection）对多个预测结果进行合成，并根据特定权重自适应地聚合输出。】
            batch_output = (gates * expert_outputs).sum(1)
            output[i, :, :] = batch_output

            importance = gates.sum(0)  # 计算重要性
            loss += self.loss_coef * self.cv_squared(importance)  # 累加损失

        # 转置输出
        output = torch.transpose(output, 0, 1)
        # 返回输出和损失
        return output, loss

if __name__ == "__main__":
    # 定义输入形状和模型参数
    input_shape = (30, 10)  # 示例：序列长度为30，特征数量为10
    pred_len = 15  # 预测长度
    # 初始化AMS模型
    model = AMS(input_shape, pred_len)
    # 示例输入数据
    batch_size = 32
    x = torch.randn(batch_size, input_shape[1], input_shape[0])  # 输入形状：[batch_size, feature_num, seq_len]
    print("输入形状:", x.shape)
    time_embedding = torch.randn(batch_size, input_shape[1], input_shape[0])  # 时间嵌入
    # 运行前向传递
    output, loss = model(x, time_embedding)
    # 打印输出和损失
    print("输出形状:", output.shape)
    print("损失:", loss.item())
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")