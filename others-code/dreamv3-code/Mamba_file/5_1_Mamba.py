import torch
#from mamba_ssm_self.modules.mamba_simple import Mamba #从我们的mamba目录导入
from mamba_ssm import Mamba

# 环境: Cuda 11.8, python 3.8(ubuntu20.04), PyTorch  2.0.0

### 不稳定安装方法
# 运气好的话,一次性安装完成,运气不好,一天也安装不好, 因为是从github直接拉取资源,非常不稳定: pip install mamba-ssm --timeout=200
### 稳定安装方法
# 1. 通过此命令行查看安装的是哪个wheel文件:pip install mamba-ssm --no-cache-dir --verbose
# 2. 复制给定的.wheel链接到浏览器,直接下载
# 3. 然后在对应的环境中直接pip install mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl


# (B,L,D)
batch, length, dim = 2, 196, 64
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
print(y.shape)
assert y.shape == x.shape