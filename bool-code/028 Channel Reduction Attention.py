import torch
import torch.nn as nn

"""
论文地址：https://openaccess.thecvf.com/content/WACV2024/papers/Kang_MetaSeg_MetaFormer-Based_Global_Contexts-Aware_Network_for_Efficient_Semantic_Segmentation_WACV_2024_paper.pdf
论文题目：MetaSeg: MetaFormer-based Global Contexts-aware Network for Efﬁcient Semantic Segmentation (WACV 2024)
中文题目：MetaSeg：基于MetaFormer的用于高效的语义分割全局上下文感知网络
讲解视频：https://www.bilibili.com/video/BV1FzmTY7Edc/
         通道缩减自注意力（Channel Reduction Self-Attention）：
         设计理念：将Q和K的通道维度简化为一维，
         作用：从而有效地考虑全局性从而实现自注意力操作的计算减少，兼顾全局上下文提取和自注意力在语义分割中的计算效率。
"""

class Channel_Reduction_SelfAttention(nn.Module):
    # 初始化函数，定义 输入通道数in_channels和 减少比率reduction_ratio，默认为16，案例为分别为 64 和 4
    def __init__(self, in_channels, reduction_ratio=16):
        super(Channel_Reduction_SelfAttention, self).__init__()
        # 计算减少后的通道数
        reduced_channels = in_channels // reduction_ratio

        # 定义用于将原始通道数映射到减少后通道数的线性层（Q投影）
        self.query_projection = nn.Linear(in_channels, reduced_channels)
        # 定义用于将原始通道数映射到减少后通道数的线性层（K投影）
        self.key_projection = nn.Linear(in_channels, reduced_channels)
        # 定义用于保持原始通道数不变的线性层（V投影）
        self.value_projection = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        # 获取输入张量的尺寸信息：批次大小、通道数、高度和宽度
        ## input = torch.randn(8, 64, 32, 32)
        #  batch_size=8 channels=64 height=32 width=32
        batch_size, channels, height, width = x.size()

        # 将输入张量展平成二维矩阵形式
        input_flat = x.view(batch_size, channels, -1)   # torch.Size([8, 64, 1024])
        # 对展平后的数据在最后一个维度上取平均，得到每个通道上的平均值
        avg_pool = torch.mean(input_flat, dim=-1, keepdim=True) # torch.Size([8, 64, 1])

        # Q: torch.Size([8, 64, 1024]) ---> torch.Size([8, 1024, 64]) --->torch.Size([8, 1024, 16]) 【除以4】
        query = self.query_projection(input_flat.permute(0, 2, 1))
        # K: torch.Size([8, 64, 1]) ---> torch.Size([8, 1, 64]) ---> torch.Size([8, 1, 16]) 【除以4】
        key = self.key_projection(avg_pool.permute(0, 2, 1))
        # V: torch.Size([8, 64, 1]) ---> torch.Size([8, 1, 64]) ---> torch.Size([8, 1, 64]) 【不变】
        value = self.value_projection(avg_pool.permute(0, 2, 1))

        # 通过矩阵乘法来计算注意力图，使用softmax确保权重之和为1
        """
            Q:torch.Size([8, 1024, 16])
            K:torch.Size([8, 1, 16]) ---> torch.Size([8, 16, 1]) 
        """
        attention_map = torch.softmax(torch.bmm(query, key.permute(0, 2, 1)), dim=1) # torch.Size([8, 1024, 1])

        # 通过矩阵乘法来计算（注意力图与V），得到最终输出
        """
            attention_map: torch.Size([8, 1024, 1])
            value        : torch.Size([8, 1, 64])
        """
        out = torch.bmm(attention_map, value)   # torch.Size([8, 1024, 64])

        # 将输出重新塑形回原始输入的形状
        ## 我认为应该input_flat.permute(0, 2, 1) 但是作者没有，因为64为通道数，1024为H与W的乘积
        out = out.view(batch_size, channels, height, width)  # 还原成原始形状 (8, 64, 32, 32)
        return out

if __name__ == "__main__":
    input = torch.randn(8, 64, 32, 32)
    CRA = Channel_Reduction_SelfAttention(in_channels=64, reduction_ratio=4)
    output = CRA(input)

    print('input_size:', input.size())
    print('output_size:', output.size())

    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")