import torch
import torch.nn as nn
import math

'''
来自于AAAI 2025 顶会    持续分享顶会顶刊模块，助力交流群里小伙伴高效去发小论文，顺利毕业！！！

即插即用模块： DRM 防御优化模块  特征优化模块/特征增强模块 （大家有没有发现顶会论文的模块取名都很有意思！）

教大家用我的五步法，学会书写顶会摘要，提高自己论文中稿率！

红外和可见光图像融合 （IVIF） 是一种通过将来自不同模态的独特信息集成到一个融合图像中来提高视觉性能的关键技术。 --—第一步，交代本文任务主题：红外和可见光图像融合

现有的方法更注重使用未受干扰的数据进行融合，而忽视了故意干扰对融合结果有效性的影响。 ---第二步，交代现有方法存在不足

为了研究融合模型的鲁棒性，在本文中，我们提出了一种新型的对抗性攻击弹性网络。 ---第三步，交代了本文创新点

具体来说，我们开发了一个具有抗攻击损失函数的对抗范式来实现对抗性攻击和训练。它是基于 IVIF 的内在本质构建的，为未来的研究进展提供了坚实的基础。
在这种范式下，我们采用 Unet 作为管道，并使用基于 transformer 的防御优化模块 （DRM），以稳健的粗到精方式保证融合图像质量。
                                                              --- 第四步，简答描述具体带来的创新点

与以前的工作相比，我们的方法减轻了对抗性扰动的不利影响，始终保持高保真融合结果。   ---第五步，对比实验，消融实验等验证本文创新点的有效性
此外，下游任务的性能也可以在对抗性攻击下得到很好的维护。

适用于：目标检测，图像融合，图像分割，语义分割，遥感图像任务，超分辨率图像，图像恢复，暗光增强等所有计算机视觉CV任务通用的即插即用模块。

'''
class PatchEmbed(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x
class PatchUnEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x

#ARB
class ESSAttn(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        b, N, C = x.shape
        qkv = self.lnqkv(x)
        qkv = torch.split(qkv, C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)
        t1 = v
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)

        attn = t1 + t2
        attn = self.ln(attn)
        return attn

    def is_same_matrix(self, m1, m2):
        rows, cols = len(m1), len(m1[0])
        for i in range(rows):
            for j in range(cols):
                if m1[i][j] != m2[i][j]:
                    return False
        return True

class Downsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1))
                m.append(nn.PixelUnshuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, num_feat // 9, 3, 1, 1))
            m.append(nn.PixelUnshuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Downsample, self).__init__(*m)
class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
class Convdown(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        self.convd = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim, 1, 1, 0))

        self.attn = ESSAttn(dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        shortcut = x
        x_size = (x.shape[2], x.shape[3])
        x_embed = self.patch_embed(x)
        x_embed = self.attn(self.norm(x_embed))  # + x_embed
        x = self.drop(self.patch_unembed(x_embed, x_size))
        x = torch.cat((x, shortcut), dim=1)
        x = self.convd(x)
        x = x + shortcut
        return x
class Convup(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        self.convu = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim, 1, 1, 0))
        self.drop = nn.Dropout2d(0.2)
        self.norm = nn.LayerNorm(dim)
        self.attn = ESSAttn(dim)

    def forward(self, x):
        shortcut = x
        x_size = (x.shape[2], x.shape[3])
        x_embed = self.patch_embed(x)
        x_embed = self.attn(self.norm(x_embed))
        x = self.drop(self.patch_unembed(x_embed, x_size))
        x = torch.cat((x, shortcut), dim=1)
        x = self.convu(x)
        x = x + shortcut
        return x
class DRM(nn.Module):
    def __init__(self, dim, upscale=1):
        super(DRM, self).__init__()
        self.convup = Convup(dim)
        self.convdown = Convdown(dim)
        self.convupsample = Upsample(scale=upscale, num_feat=dim)
        self.convdownsample = Downsample(scale=upscale, num_feat=dim)

    def forward(self, x):
        xup = self.convupsample(x)
        x1 = self.convup(xup) # 就是图中的A注意力模块
        xdown = self.convdownsample(x1) + x
        x2 = self.convdown(xdown) #就是图中的A注意力模块
        xup = self.convupsample(x2) + x1
        x3 = self.convup(xup)
        xdown = self.convdownsample(x3) + x2
        x4 = self.convdown(xdown)
        xup = self.convupsample(x4) + x3
        x5 = self.convup(xup)
        return x5

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1, 64, 128, 128)
    drm = DRM(64)
    output = drm(input)
    print("DRM_input.shape:", input.shape)
    print("DRM_output.shape:",output.shape)

