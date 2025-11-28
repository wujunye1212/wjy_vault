from typing import Optional, Callable, Union, Tuple, Any
import torch
from torch import nn, Tensor
import numpy as np
from typing import Optional
import math
from torch import nn
# 代码：https://github.com/anthonyweidai/SvANet/tree/main
# 论文：https://arxiv.org/pdf/2407.07720
'''
SvANet：一种用于小型医疗对象分割的基于尺度变异注意力的网络
即插即用注意力模块：蒙特卡洛注意力（MCAttention）
摘要—-早期发现和准确诊断可以预测恶性疾病转化的风险，从而增加有效治疗的可能性。 

感染区域小的轻度综合征是一个不祥的警告，在疾病的早期诊断中是最重要的。
深度学习算法，如卷积神经网络 （CNN），已被用于医学图像分割，显示出有希望的结果。
然而，由于 CNN 中的卷积和池化操作会导致信息丢失和压缩缺陷，
因此分析图像中小区域的医疗物体仍然是一个挑战。随着网络的深入，
这些损失和缺陷变得越来越严重，特别是对于小型医疗对象。为了应对这些挑战，
我们提出了一种新的基于尺度变化注意力的网络（SvANet），用于医学图像中准确的小尺度目标分割。

SvANet 由蒙特卡洛注意力（MCAttention）、尺度变异注意力和视觉转换器组成，
它结合了跨尺度特征并减轻了压缩伪影，以增强对小型医疗物体的区分。
定量实验结果表明，在小尺度目标分割方面效果比较好，在分割肾肿瘤、皮损、肝肿瘤、息肉、手术切除细胞、视网膜血管系统方面。

适用于：小目标图像分割任务，小目标检测任务可以优先考虑。其他CV图像任务通用注意力模块
'''
def makeDivisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
def callMethod(self, ElementName):
    return getattr(self, ElementName)
def setMethod(self, ElementName, ElementValue):
    return setattr(self, ElementName, ElementValue)
def shuffleTensor(Feature: Tensor, Mode: int=1) -> Tensor:
    # shuffle multiple tensors with the same indexs
    # all tensors must have the same shape
    if isinstance(Feature, Tensor):
        Feature = [Feature]

    Indexs = None
    Output = []
    for f in Feature:
        # not in-place operation, should update output
        B, C, H, W = f.shape
        if Mode == 1:
            # fully shuffle
            f = f.flatten(2)
            if Indexs is None:
                Indexs = torch.randperm(f.shape[-1], device=f.device)
            f = f[:, :, Indexs.to(f.device)]
            f = f.reshape(B, C, H, W)
        else:
            # shuflle along y and then x axis
            if Indexs is None:
                Indexs = [torch.randperm(H, device=f.device),
                          torch.randperm(W, device=f.device)]
            f = f[:, :, Indexs[0].to(f.device)]
            f = f[:, :, :, Indexs[1].to(f.device)]
        Output.append(f)
    return Output
class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size: int or tuple=1):
        super(AdaptiveAvgPool2d, self).__init__(output_size=output_size)

    def profileModule(self, Input: Tensor):
        Output = self.forward(Input)
        return Output, 0.0, 0.0

class AdaptiveMaxPool2d(nn.AdaptiveMaxPool2d):
    def __init__(self, output_size: int or tuple=1):
        super(AdaptiveMaxPool2d, self).__init__(output_size=output_size)

    def profileModule(self, Input: Tensor):
        Output = self.forward(Input)
        return Output, 0.0, 0.0
class BaseConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: Optional[int] = 1,
            padding: Optional[int] = None,
            groups: Optional[int] = 1,
            bias: Optional[bool] = None,
            BNorm: bool = False,
            # norm_layer: Optional[Callable[..., nn.Module]]=nn.BatchNorm2d,
            ActLayer: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
            Momentum: Optional[float] = 0.1,
            **kwargs: Any
    ) -> None:
        super(BaseConv2d, self).__init__()
        if padding is None:
            padding = int((kernel_size - 1) // 2 * dilation)

        if bias is None:
            bias = not BNorm

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias

        self.Conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, groups, bias, **kwargs)

        self.Bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=Momentum) if BNorm else nn.Identity()

        if ActLayer is not None:
            if isinstance(list(ActLayer().named_modules())[0][1], nn.Sigmoid):
                self.Act = ActLayer()
            else:
                self.Act = ActLayer(inplace=True)
        else:
            self.Act = ActLayer

        self.apply(initWeight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x

NormLayerTuple = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.GroupNorm,
    nn.BatchNorm3d,
)
def initWeight(Module):
    # init conv, norm , and linear layers
    ## empty module
    if Module is None:
        return
    ## conv layer
    elif isinstance(Module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(Module.weight, a=math.sqrt(5))
        if Module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(Module.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(Module.bias, -bound, bound)
    ## norm layer
    elif isinstance(Module, NormLayerTuple):
        if Module.weight is not None:
            nn.init.ones_(Module.weight)
        if Module.bias is not None:
            nn.init.zeros_(Module.bias)
    ## linear layer
    elif isinstance(Module, nn.Linear):
        nn.init.kaiming_uniform_(Module.weight, a=math.sqrt(5))
        if Module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(Module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(Module.bias, -bound, bound)
    elif isinstance(Module, (nn.Sequential, nn.ModuleList)):
        for m in Module:
            initWeight(m)
    elif list(Module.children()):
        for m in Module.children():
            initWeight(m)
class MCAttention(nn.Module):
    # Monte carlo attention
    def __init__(
            self,
            InChannels: int,
            HidChannels: int = None,
            SqueezeFactor: int = 4,
            PoolRes: list = [1, 2, 3],
            Act: Callable[..., nn.Module] = nn.ReLU,
            ScaleAct: Callable[..., nn.Module] = nn.Sigmoid,
            MoCOrder: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__()
        if HidChannels is None:
            HidChannels = max(makeDivisible(InChannels // SqueezeFactor, 8), 32)

        AllPoolRes = PoolRes + [1] if 1 not in PoolRes else PoolRes
        for k in AllPoolRes:
            Pooling = AdaptiveAvgPool2d(k)
            setMethod(self, 'Pool%d' % k, Pooling)

        self.SELayer = nn.Sequential(
            BaseConv2d(InChannels, HidChannels, 1, ActLayer=Act),
            BaseConv2d(HidChannels, InChannels, 1, ActLayer=ScaleAct),
        )

        self.PoolRes = PoolRes
        self.MoCOrder = MoCOrder

    def monteCarloSample(self, x: Tensor) -> Tensor:
        if self.training:
            PoolKeep = np.random.choice(self.PoolRes)
            x1 = shuffleTensor(x)[0] if self.MoCOrder else x
            AttnMap: Tensor = callMethod(self, 'Pool%d' % PoolKeep)(x1)
            if AttnMap.shape[-1] > 1:
                AttnMap = AttnMap.flatten(2)
                AttnMap = AttnMap[:, :, torch.randperm(AttnMap.shape[-1])[0]]
                AttnMap = AttnMap[:, :, None, None]  # squeeze twice
        else:
            AttnMap: Tensor = callMethod(self, 'Pool%d' % 1)(x)

        return AttnMap

    def forward(self, x: Tensor) -> Tensor:
        AttnMap = self.monteCarloSample(x)
        return x * self.SELayer(AttnMap)


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    model = MCAttention(InChannels=64, HidChannels=16)
    input = torch.randn(1, 64, 32, 32)
    output = model(input)
    print('input_size:',input.size())
    print('output_size:',output.size())