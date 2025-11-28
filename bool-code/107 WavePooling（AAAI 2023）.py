import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
# pip install timm == 1.0.8

"""
    论文地址：https://ojs.aaai.org/index.php/AAAI/article/download/25592/25364
    论文题目：Wave Superposition Inspired Pooling for Dynamic Interactions-Aware Trajectory Prediction (AAAI 2023)
    中文题目：基于波叠加的动态相互作用感知轨迹预测池
    讲解视频：https://www.bilibili.com/video/BV13dPkegEYU
        波形池化方法/相位注意力变换模块（Wave-Pooling）：
            解决问题：自动驾驶系统中的路径规划和避免碰撞，需要考虑多辆车之间动态交互的复杂道路环境。   
            实现方式：该机制受Wave-MLP启发，将每个车辆视为一个波，其中振幅反映车辆的动力学特性，而相位则调节车辆间的相互作用，通过波叠加可以反映出车辆之间的动态交互，用于更有效地捕捉动态和高阶交互，并预测周围车辆的运动轨迹。
            涨点以后如何写作？：【此部分 请务必看视频 视频更为详细】
"""

class Mlp(nn.Module):  # 定义一个多层感知机类
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):  # 构造函数
        super().__init__()  # 初始化父类
        out_features = out_features or in_features  # 输出特征数默认等于输入特征数
        hidden_features = hidden_features or in_features  # 隐藏层特征数默认等于输入特征数
        self.act = act_layer()  # 激活层
        self.drop = nn.Dropout(drop)  # Dropout 层
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)  # 第一个卷积层
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)  # 第二个卷积层

    def forward(self, x):  # 前向传播函数
        x = self.fc1(x)  # 输入通过第一个卷积层
        x = self.act(x)  # 通过激活层
        x = self.drop(x)  # 通过 Dropout 层
        x = self.fc2(x)  # 输入通过第二个卷积层
        x = self.drop(x)  # 再次通过 Dropout 层
        return x  # 返回处理后的张量

class PATM(nn.Module):  # 定义相位注意力变换模块类
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mode='fc'):  # 构造函数
        super().__init__()  # 初始化父类

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)  # 用于提取水平方向的特征
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)  # 用于提取垂直方向的特征

        self.fc_c = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)  # 用于提取通道方向的特征
        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), stride=1, padding=(0, 7 // 2), groups=dim, bias=False)  # 用于处理水平方向的特征
        self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), stride=1, padding=(7 // 2, 0), groups=dim, bias=False)  # 用于处理垂直方向的特征
        self.reweight = Mlp(dim, dim // 4, dim * 3)  # 用于重新加权
        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)  # 投影层z
        self.proj_drop = nn.Dropout(proj_drop)  # Dropout 层
        self.mode = mode  # 模式选择
        if mode == 'fc':  # 如果模式是全连接
            self.theta_h_conv = nn.Sequential(  # 水平方向相位计算
                nn.Conv2d(dim, dim, 1, 1, bias=True),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            )
            self.theta_w_conv = nn.Sequential(  # 垂直方向相位计算
                nn.Conv2d(dim, dim, 1, 1, bias=True),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            )
        else:  # 如果不是全连接模式
            self.theta_h_conv = nn.Sequential(  # 水平方向相位计算
                nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            )
            self.theta_w_conv = nn.Sequential(  # 垂直方向相位计算
                nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            )

    def forward(self, x):  # 前向传播函数
        B, C, H, W = x.shape  # 获取输入张量的维度

        theta_h = self.theta_h_conv(x)  # 计算水平方向的相位
        theta_w = self.theta_w_conv(x)  # 计算垂直方向的相位
        x_h = self.fc_h(x)  # 提取水平方向的振幅
        x_w = self.fc_w(x)  # 提取垂直方向的振幅

        # 【创新点 欧拉公式的特征加权】
        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)  # 欧拉公式展开水平方向
        x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)  # 欧拉公式展开垂直方向

        h = self.tfc_h(x_h)  # 处理水平方向的特征 H
        w = self.tfc_w(x_w)  # 处理垂直方向的特征 W
        c = self.fc_c(x)  # 提取通道方向的特征 C

        # 【取权重】
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)  # 平均池化
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)  # 重新加权
        x = h * a[0] + w * a[1] + c * a[2]  # 加权组合

        x = self.proj(x)  # 通过投影层
        x = self.proj_drop(x)  # 通过 Dropout 层
        return x  # 返回处理后的张量

class WaveBlock(nn.Module):  # 定义波形块类
    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):  # 构造函数
        super().__init__()  # 初始化父类
        self.norm1 = norm_layer(dim)  # 第一个规范化层
        self.attn = PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop, mode=mode)  # 相位注意力变换模块
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # DropPath 层
        self.norm2 = norm_layer(dim)  # 第二个规范化层
        mlp_hidden_dim = int(dim * mlp_ratio)  # MLP 的隐藏层维度
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)  # 多层感知机模块

    def forward(self, x):  # 前向传播函数
        x = x + self.drop_path(self.attn(self.norm1(x)))  # 注意力变换
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # MLP 变换
        return x  # 返回处理后的张量

if __name__ == '__main__':  # 主程序入口
    block = WaveBlock(dim=64)  # 假设输入特征的通道数为 64
    input = torch.rand(10, 64, 32, 32)  # 假设输入大小为 (batch_size=10, channels=64, height=32, width=32)
    # 运行前向传播
    output = block(input)  # 调用波形块的前向传播
    # 打印输入和输出的大小
    print("输入大小:", input.size())  # 打印输入张量的大小
    print("输出大小:", output.size())  # 打印输出张量的大小
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")