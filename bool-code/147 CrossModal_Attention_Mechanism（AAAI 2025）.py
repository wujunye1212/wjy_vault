import torch
from torch import nn

"""
    论文地址：https://ojs.aaai.org/index.php/AAAI/article/view/32157/34312
    论文题目：SalM2: An Extremely Lightweight Saliency Mamba Model for Real-Time Cognitive Awareness of Driver Attention（AAAI 2025）
    中文题目：SalM2：一种用于驾驶员注意力实时认知感知的超轻量级显著性曼巴模型（AAAI 2025）
    讲解视频：https://www.bilibili.com/video/BV1uaLkzxEdi/
        跨模态注意力机制（Cross Modal Attention Mechanism, CAM）：
            实际意义：①模态信息融合难：在自动驾驶中，需结合场景语义信息（限速）和图像信息（行人）。然而，文本数据具有很强语义和逻辑性，与图像数据在特征空间中难以匹配和对齐。
                     ②特征维度不一致问题：传统语义信息和图像信息的特征维度存在差异，两者难以直接融合。
            实现方式：交叉注意力的混合使用。（不仅限于文本、2D图像、3D图像、灰度图、红外图像）
            产生思考：在交叉学科/研究方向中，其实并不在意你的模型是否有创新，与实际问题对应解决，就是好文章。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】   
"""
# Cross-Modal Attention Mechanism（跨模态注意力机制，简称 CMA）
class CAM(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个可学习的参数gamma，初始值为0
        self.gamma = nn.Parameter(torch.zeros(1))
        # 定义一个Softmax层，用于在最后一个维度上进行softmax操作
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img_feat, text_feat):
        # 获取图像特征张量的形状，B为批次大小，C为通道数，H为高度，W为宽度
        B, C, H, W = img_feat.shape

        # 将图像特征张量进行变形，将后两个维度展平，方便后续矩阵乘法操作
        q = img_feat.view(B, C, -1)
        # 将文本特征张量进行变形，展平后交换最后两个维度，以便后续与q进行矩阵乘法
        k = text_feat.view(B, C, -1).permute(0, 2, 1)
        # 计算图像特征和文本特征的注意力映射，通过矩阵乘法得到
        attention_map = torch.bmm(q, k)
        # 对注意力映射进行softmax操作，使其值在0到1之间且总和为1
        attention_map = self.softmax(attention_map)

        # 将文本特征张量进行变形，展平以便后续与注意力映射进行矩阵乘法
        v = text_feat.view(B, C, -1)
        # 通过注意力映射对文本特征进行加权，得到注意力信息
        attention_info = torch.bmm(attention_map, v)
        # 将注意力信息的形状恢复为与输入图像特征相同的形状
        attention_info = attention_info.view(B, C, H, W)
        # 将注意力信息与图像特征进行加权融合，得到最终输出
        output = self.gamma * attention_info + img_feat
        return output

if __name__ == "__main__":
    # 图像特征张量，批次大小为1，通道数为32，高度和宽度为50
    img_feat = torch.randn(1, 32, 50, 50)
    # 文本特征张量，批次大小为1，通道数为32，高度和宽度为50
    text_feat = torch.randn(1, 32, 50, 50)
    model = CAM()
    output = model(img_feat, text_feat)
    print(f'img_Input size: {img_feat.size()}')
    print(f'text_Input size: {text_feat.size()}')
    print(f'Output size: {output.size()}')
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")