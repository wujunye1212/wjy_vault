import torch
import torch.nn as nn
from einops import rearrange

"""
    论文地址：https://arxiv.org/pdf/2407.05128
    论文题目：SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention（2025 JCR一区TOP）
    中文题目：SCSA：探索空间注意力和通道注意力之间的协同效应（2025 JCR一区TOP）
    讲解视频：https://www.bilibili.com/video/BV1DJLLzUE8r/
    渐进式通道自注意力（Progressive Channel-wise Self-Attention, PCSA）：
        实际意义：①传统卷积计算通道注意力的不足问题：传统方法常用卷积运算探索通道间依赖关系来计算通道注意力，但这种方式不够直观，难以有效衡量不同通道间的相似性。
                ②多语义信息处理问题：不同子特征在多语义信息上存在差异，这会影响模型对整体语义的理解和信息融合效果。
        实现方式：①用平均池化处理特征图，降分辨率、减计算量。
                ②沿通道维度用单头自注意力机制，经 Softmax 得注意力权重，加权求和获自注意力特征。
                ③通过Sigmoid 生成权重与原特征图相乘，优化通道特征。
        涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】   
"""
class Progressive_Channel_wise_Self_Attention(nn.Module):
    def __init__(
            self,
            dim: int,  # 输入特征的维度
            head_num: int = 8,  # 注意力头的数量
            qkv_bias: bool = False,  # 是否使用qkv的偏置
            attn_drop_ratio: float = 0.,  # 注意力丢弃率
            gate_layer: str = 'sigmoid',  # 门控层的激活函数类型
    ):
        super(Progressive_Channel_wise_Self_Attention, self).__init__()
        self.dim = dim  # 保存输入特征的维度
        self.head_num = head_num  # 保存注意力头的数量
        self.head_dim = dim // head_num  # 计算每个注意力头的维度
        self.scaler = self.head_dim ** -0.5  # 缩放因子，用于注意力计算

        # 断言输入特征的维度应能被4整除
        assert self.dim % 4 == 0, '输入特征的维度应能被4整除。'

        # 定义组归一化层
        self.norm = nn.GroupNorm(1, dim)
        # 定义查询（Q）卷积层
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        # 定义键（K）卷积层
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        # 定义值（V）卷积层
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        # 定义注意力丢弃层
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        # 根据门控层类型选择激活函数用于通道注意力
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        temp = y
        y = y.mean((2, 3), keepdim=True)
        _, _, h_, w_ = y.size()
        y = self.norm(y)

        # 计算查询（Q）
        q = self.q(y)
        # 计算键（K）
        k = self.k(y)
        # 计算值（V）
        v = self.v(y)
        # 调整Q、K、V的形状
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        # 计算注意力分数
        attn = q @ k.transpose(-2, -1) * self.scaler
        # 对注意力分数进行归一化并丢弃部分注意力
        attn = self.attn_drop(attn.softmax(dim=-1))
        # 计算注意力加权后的特征
        attn = attn @ v
        # 调整注意力加权后特征的形状
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))

        # 在高度和宽度维度上求平均
        attn = attn.mean((2, 3), keepdim=True)
        # 对注意力进行通道注意力门控
        attn = self.ca_gate(attn)
        # 将通道注意力应用到输入特征上
        return attn * temp

if __name__ == '__main__':
    model = Progressive_Channel_wise_Self_Attention(dim=64)  # 创建SCSA模块实例
    input = torch.randn(1, 64, 50, 50)  # 生成随机输入张量
    output = model(input)  # 前向传播计算输出
    print(f'Input size: {input.size()}')  # 打印输入张量的大小
    print(f'Output size: {output.size()}')  # 打印输出张量的大小
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")