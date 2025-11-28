import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# https://arxiv.org/pdf/2410.05258
''' 
来自清华大学 Arxiv 2024     
即插即用模块：MHDA 多头差分注意力模块   
所有NLP和CV任务通用的即插即用注意力模块

Transformer 模型倾向于将注意力过多分配到与任务无关的上下文，导致注意力噪声问题。
在此工作中，作者提出了一种名为 DIFF Transformer 的新架构，使用差分注意力机制来强化对关键信息的关注，
同时抑制噪声。具体而言，该机制通过计算两组独立的 Softmax 注意力分布之间的差值来实现降噪，从而促进稀疏注意力模式的形成。

实验表明，DIFF Transformer 在扩展模型规模和训练数据时的表现优于传统 Transformer，
并在长上下文建模、关键信息检索、减少幻觉（如问答和文本摘要任务中的错误生成）以及上下文学习等实际应用中表现出显著优势。
此外，该方法还能减轻模型激活值的离群问题，为量化和训练稳定性提供新的可能性。
通过更专注于相关上下文，DIFF Transformer 提高了模型的准确性和鲁棒性，
尤其是在处理长序列和复杂任务时。

核心创新：
引入了差分注意力机制，通过计算两个独立的 Softmax 注意力分布的差异来消除噪声，增强对关键信息的关注。
它确保模型将注意力集中在与任务相关的关键信息上，而不是被无关上下文干扰。
类似于降噪耳机中的差分放大器，该机制通过相减操作过滤掉公共噪声。
实验结果：
DIFF Transformer 在语言建模、长序列建模、信息检索、幻觉减轻
和上下文学习等任务中均优于传统 Transformer。
在参数量和训练数据减少的情况下，仍然可以达到相似的性能。

'''
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Applies normalization across the last dimension and scales the output.
    """

    def __init__(self, d, eps=1e-5):
        """
        Args:
            d (int): Dimension of the input features.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        """
        Forward pass for RMSNorm.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d).

        Returns:
            Tensor: Normalized and scaled tensor.
        """
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale
class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function.
    Combines the Swish activation with Gated Linear Units.
    """

    def __init__(self, d_model):
        """
        Args:
            d_model (int): Dimension of the input features.
        """
        super().__init__()
        # Intermediate projection layers
        # Typically, SwiGLU splits the computation into two parts
        self.WG = nn.Linear(d_model, d_model * 2)
        self.W1 = nn.Linear(d_model, d_model * 2)
        self.W2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        """
        Forward pass for SwiGLU.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after applying SwiGLU.
        """
        # Apply the gates
        g = F.silu(self.WG(x))  # Activation part
        z = self.W1(x)  # Linear part
        # Element-wise multiplication and projection
        return self.W2(g * z)
class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention Mechanism.
    Replaces the conventional softmax attention with a differential attention.
    Incorporates a causal mask to ensure autoregressive behavior.
    """

    def __init__(self, d_model, num_heads, lambda_init):
        """
        Args:
            d_model (int): Dimension of the model. Must be divisible by num_heads.
            num_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear projections for queries, keys, and values
        # Project to 2 * d_head per head for differential attention
        self.W_q = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_o = nn.Linear(2 * self.d_head * num_heads, d_model, bias=False)

        # Learnable parameters for lambda reparameterization
        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.d_head))

        self.lambda_init = lambda_init

        # Scale parameter for RMSNorm
        self.rms_scale = nn.Parameter(torch.ones(2 * self.d_head))
        self.eps = 1e-5  # Epsilon for numerical stability

        # Initialize weights (optional but recommended)
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters for improved training stability.
        """
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.constant_(self.rms_scale, 1.0)

    def forward(self, X):
        """
        Forward pass for Multi-Head Differential Attention.

        Args:
            X (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after applying differential attention.
        """
        batch, N, d_model = X.shape

        # Project inputs to queries, keys, and values
        Q = self.W_q(X)  # Shape: (batch, N, 2 * num_heads * d_head)
        K = self.W_k(X)  # Shape: (batch, N, 2 * num_heads * d_head)
        V = self.W_v(X)  # Shape: (batch, N, 2 * num_heads * d_head)

        # Reshape and permute for multi-head attention
        # New shape: (batch, num_heads, sequence_length, 2 * d_head)
        Q = Q.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        K = K.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        V = V.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)

        # Split Q and K into Q1, Q2 and K1, K2
        Q1, Q2 = Q.chunk(2, dim=-1)  # Each of shape: (batch, num_heads, N, d_head)
        K1, K2 = K.chunk(2, dim=-1)  # Each of shape: (batch, num_heads, N, d_head)

        # Compute lambda using reparameterization
        # lambda_val = exp(lambda_q1 . lambda_k1) - exp(lambda_q2 . lambda_k2) + lambda_init
        # Compute dot products for each head
        # Shape of lambda_val: (num_heads,)
        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()  # (num_heads,)
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()  # (num_heads,)
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init  # (num_heads,)

        # Expand lambda_val to match attention dimensions
        # Shape: (batch, num_heads, 1, 1)
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # ------------------- Causal Mask Implementation ------------------- #
        # Create a causal mask to prevent attention to future tokens
        # Shape of mask: (1, 1, N, N)
        mask = torch.tril(torch.ones((N, N), device=X.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        # Replace 1s with 0.0 and 0s with -inf
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        # -------------------------------------------------------------------- #

        # Compute attention scores
        scaling = 1 / math.sqrt(self.d_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling  # (batch, num_heads, N, N)
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling  # (batch, num_heads, N, N)

        # Apply the causal mask
        A1 = A1 + mask  # Mask out future positions
        A2 = A2 + mask  # Mask out future positions

        # Apply softmax to get attention weights
        attention1 = F.softmax(A1, dim=-1)  # (batch, num_heads, N, N)
        attention2 = F.softmax(A2, dim=-1)  # (batch, num_heads, N, N)
        attention = attention1 - lambda_val * attention2  # (batch, num_heads, N, N)

        # Apply attention weights to values
        O = torch.matmul(attention, V)  # (batch, num_heads, N, 2 * d_head)

        # Normalize each head independently using RMSNorm
        # First, reshape for RMSNorm
        O_reshaped = O.contiguous().view(batch * self.num_heads, N, 2 * self.d_head)  # (batch*num_heads, N, 2*d_head)

        # Compute RMSNorm
        rms_norm = torch.sqrt(O_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (batch*num_heads, N, 1)
        O_normalized = (O_reshaped / rms_norm) * self.rms_scale  # (batch*num_heads, N, 2*d_head)

        # Reshape back to (batch, num_heads, N, 2 * d_head)
        O_normalized = O_normalized.view(batch, self.num_heads, N, 2 * self.d_head)

        # Scale the normalized output
        O_normalized = O_normalized * (1 - self.lambda_init)  # Scalar scaling

        # Concatenate all heads
        # New shape: (batch, N, num_heads * 2 * d_head)
        O_concat = O_normalized.transpose(1, 2).contiguous().view(batch, N, self.num_heads * 2 * self.d_head)

        # Final linear projection
        out = self.W_o(O_concat)  # (batch, N, d_model)

        return out


class DiffTransformerLayer(nn.Module):
    """
    Single Layer of the DiffTransformer Architecture.
    Consists of Multi-Head Differential Attention followed by a SwiGLU Feed-Forward Network.
    """

    def __init__(self, d_model, num_heads, lambda_init):
        """
        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda in Differential Attention.
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadDifferentialAttention(d_model, num_heads, lambda_init)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model)

    def forward(self, x):
        """
        Forward pass for a single transformer layer.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after processing through the layer.
        """
        # Apply Multi-Head Differential Attention with residual connection
        y = self.attn(self.norm1(x)) + x
        # Apply SwiGLU Feed-Forward Network with residual connection
        z = self.ff(self.norm2(y)) + y
        return z

# 输入 B  L C ,  输出 B L C
if __name__ == '__main__':
    # 初始化MultiHeadDifferentialAttention模型
    MHDA = MultiHeadDifferentialAttention(d_model=512, num_heads=8, lambda_init=0.8)

    # 1.如何输入的是图片4维数据 . CV方向的小伙伴都可以拿去使用
    # 随机生成输入4维度张量：B, C, H, W
    input_img = torch.randn(1, 512, 32, 32)
    input = input_img.reshape(1, 512, -1).transpose(-1, -2)  # B L C :1 1024 512
    # 运行前向传递
    output = MHDA(input)
    output_img = output.view(1, 512, 32, 32)  # 将三维度转化成图片四维度张量
    # 输出输入图片张量和输出图片张量的形状
    print("CV_MHDA_input size:", input_img.size())
    print("CV_MHDA_Output size:", output_img.size())

    # 2.如何输入的3维数据 . NLP方向的小伙伴都可以拿去使用
    B, L, C = 1, 1024, 512  # 批量大小、序列长度、特征维度
    # 创建一个随机的输入三维张量
    input = torch.randn(B, L, C)  # 输入三维张量为 (B, L, C)
    # 进行前向传播
    output = MHDA(input)
    print('NLP_MHDA_input size:', input.size())
    print('NLP_MHDA_output size:', output.size())

