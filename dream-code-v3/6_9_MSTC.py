import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels  # 设置输出通道数与输入通道数一致
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilations = dilations
        self.residual = residual
        self.residual_kernel_size = residual_kernel_size

        # 多尺度时间卷积分支
        self.branches = nn.ModuleList()
        for dilation in dilations:
            effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
            padding = (effective_kernel_size - 1) // 2  # 确保padding是整数
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), padding=(padding, 0), dilation=dilation,groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ))

        # 最大池化和1x1卷积分支用于捕获不同的特征
        self.maxpool_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0,groups=in_channels),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),  # 保持不变
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.conv1x1_branch = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0,groups=in_channels)

        # 残差连接
        if self.residual:
            if self.residual_kernel_size == 1:
                self.residual_connection = nn.Identity()
            else:
                self.residual_connection = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=(residual_kernel_size, 1), stride=stride, padding=0,groups=in_channels),
                    nn.BatchNorm2d(in_channels)
                )

        # 初始化权重
        self.apply(weights_init)

    def forward(self, x):
        residual = self.residual_connection(x) # 保存自身信息: (B,C,T,1)

        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x)) # 执行多个具有不同dilation的K=3的卷积层, 并保持输入和输出通道一致: (B,C,T,1)--branch-->(B,C,T,1)
        branch_outputs.append(self.maxpool_branch(x)) # 对输入x执行最大池化: (B,C,T,1)--maxpool-->(B,C,T,1)
        branch_outputs.append(self.conv1x1_branch(x)) # 对输入x执行1×1Conv: (B,C,T,1)--Conv1×1-->(B,C,T,1)

        # 合并所有分支的输出,并取平均值
        out = sum(branch_outputs) / len(branch_outputs)

        # 加上残差
        if self.residual:
            out += residual

        return out



if __name__ == '__main__':
    # (B,C,T,1)
    x1 = torch.randn(1,64,196,1).to(device)

    Model = MultiScale_TemporalConv(in_channels=64,
                                     kernel_size=3,
                                     stride=1,
                                     dilations=[1, 2, 3, 4],
                                     residual=True,
                                     residual_kernel_size=1).to(device)

    out = Model(x1) # (B,L,D)-->(B,L,D)
    print(out.shape)