import torch
import torch.nn as nn
# 代码： https://github.com/hyunwoo137/MetaSeg/tree/main?tab=readme-ov-file
# 论文：https://openaccess.thecvf.com/content/WACV2024/papers/Kang_MetaSeg_MetaFormer-Based_Global_Contexts-Aware_Network_for_Efficient_Semantic_Segmentation_WACV_2024_paper.pdf

'''
MetaSeg：基于 MetaFormer 的全局上下文感知网络，用于高效的语义分割    WACV 2024 顶会
通道缩减注意力即插即用模块：CRAttention

最近的分割方法表明，使用基于 CNN 的骨干网提取空间信息和使用解码器提取全局信息
比使用基于 Transformer 的骨干网和基于 CNN 的解码器更有效。
这促使我们采用使用 MetaFormer 模块的基于 CNN 的骨干网，并设计基于 MetaFormer 的解码器，
该解码器由一个新颖的自注意力模块组成，用于捕获全局上下文。

为了兼顾全局上下文提取和自注意力在语义分割中的计算效率，
我们提出了一种通道缩减注意力（CRA）模块，通过将查询和键的通道维度简化为一维，
从而有效地考虑全局性，从而实现自注意力操作的计算减少。
通过这种方式，我们提出的MetaSeg优于以前最先进的方法，

在流行的语义分割和医学图像分割上具有更高的计算效率。

适用于：语义分割，实例分割，目标检测，图像增强，暗光增强等所有CV2维任务通用注意力模块
'''
class CRA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CRA, self).__init__()
        reduced_channels = in_channels // reduction_ratio

        self.query_projection = nn.Linear(in_channels, reduced_channels)
        self.key_projection = nn.Linear(in_channels, reduced_channels)
        self.value_projection = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        input_flat = x.view(batch_size, channels, -1)

        avg_pool = torch.mean(input_flat, dim=-1, keepdim=True)

        query = self.query_projection(input_flat.permute(0, 2, 1))
        key = self.key_projection(avg_pool.permute(0, 2, 1))
        value = self.value_projection(avg_pool.permute(0, 2, 1))

        attention_map = torch.softmax(torch.bmm(query, key.permute(0, 2, 1)), dim=1)
        out = torch.bmm(attention_map, value)

        out = out.view(batch_size, channels, height, width)  # 还原成原始形状
        return out

if __name__ == "__main__":
    input = torch.randn(8, 64, 32, 32)
    CRA = CRA(in_channels=64, reduction_ratio=4)
    output = CRA(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
