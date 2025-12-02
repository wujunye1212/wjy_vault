import torch
from torch import nn

# sMLPBlock.py
# --------------------------------------------------------
# 论文: Sparse MLP for Image Recognition: Is Self-Attention Really Necessary? (AAAI 2022)
# 论文地址: https://ojs.aaai.org/index.php/AAAI/article/view/20133
# 抖音、B站、小红书、CSDN  布尔大学士
# 代码讲解：https://www.bilibili.com/video/BV1ENs9eiE2Z/
# ------

# 定义sMLPBlock类，继承自nn.Module
class sMLPBlock(nn.Module):
    # 初始化函数，定义网络结构
    def __init__(self, h=224, w=224, c=3):  # 输入参数为图像的高度h、宽度w以及通道数c
        super().__init__()  # 调用父类构造器
        self.proj_h = nn.Linear(h, h)  # 定义沿高度方向的线性变换层
        self.proj_w = nn.Linear(w, w)  # 定义沿宽度方向的线性变换层
        self.fuse = nn.Linear(3 * c, c)  # 定义融合层，用于融合来自不同方向的信息

    # 前向传播函数
    def forward(self, x):

        # 沿高度方向进行线性变换，并调整维度顺序
        # [B,C,H,W]  ---》 [B,C,W,H]
        #  因为 nn.Linear 默认对最后一个维度进行操作，所以这里先线性层后调换位置【常见套路】
        x_h = self.proj_h(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # torch.Size([50, 3, 224, 224])
        # 沿宽度方向进行线性变换
        x_w = self.proj_w(x)   # torch.Size([50, 3, 224, 224])

        # 保留原始输入作为残差连接
        x_id = x   # torch.Size([50, 3, 224, 224])

        # 在通道维度上合并不同的特征
        x_fuse = torch.cat([x_h, x_w, x_id], dim=1)   # torch.Size([50, 9, 224, 224])

        # 融合信息并调整维度顺序
        x_fuse_Total = x_fuse.permute(0, 2, 3, 1) # torch.Size([50, 224, 224, 9])
        # 因为 nn.Linear 默认对最后一个维度进行操作，所以这里先线性层后调换位置【常见套路】
        out = self.fuse(x_fuse_Total).permute(0, 3, 1, 2)   # torch.Size([50, 3, 224, 224])
        return out  # 返回处理后的输出张量

# 测试代码
if __name__ == '__main__':
    input = torch.randn(50, 3, 224, 224)  # 随机生成一批大小为50x3x224x224的输入数据
    print(input.shape) # torch.Size([50, 3, 224, 224])

    # 实例化sMLPBlock对象
    smlp = sMLPBlock(h=224, w=224)

    # 将输入数据传递给sMLPBlock进行前向传播
    out = smlp(input)
    print(out.shape) # torch.Size([50, 3, 224, 224])
    print("抖音、B站、小红书、CSDN同号")  # 输出作者的社交平台账号信息
    print("布尔大学士 提醒您：代码无误~~~~")  # 输出作者的提醒信息
