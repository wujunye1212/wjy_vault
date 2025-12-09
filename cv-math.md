# 计算机视觉的数学基础（Mathematical Foundations of Computer Vision）

---

# 1. 多维信号与张量（Tensors）

## 1.1 图像作为张量
灰度图：
$$
I \in \mathbb{R}^{H \times W}
$$

RGB 彩色图：
$$
I \in \mathbb{R}^{H \times W \times 3}
$$

视频序列：
$$
V \in \mathbb{R}^{T \times H \times W \times C}
$$

深度学习中的激活：
$$
X \in \mathbb{R}^{N \times H \times W \times C}
$$

---

# 2. 图像几何（Image Geometry）

## 2.1 仿射变换
$$
\begin{pmatrix}
x' \\ y' \\ 1
\end{pmatrix}
=
\begin{pmatrix}
a_{11} & a_{12} & t_x \\
a_{21} & a_{22} & t_y \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
x \\ y \\ 1
\end{pmatrix}
$$

## 2.2 投影变换（Homography）
$$
x' \sim H x,\quad H \in \mathbb{R}^{3\times 3}
$$

## 2.3 旋转矩阵
3D 旋转：
$$
R \in SO(3),\quad R^TR = I,\quad \det(R)=1
$$

绕轴旋转（Rodrigues 公式）：
$$
R = I + \sin\theta [v]_\times + (1-\cos\theta)[v]_\times^2
$$

---

# 3. 相机成像模型（Camera Model）

## 3.1 透视投影
$$
s
\begin{pmatrix}
u\\v\\1
\end{pmatrix}
=
K[R|t]
\begin{pmatrix}
X\\Y\\Z\\1
\end{pmatrix}
$$

## 3.2 内参矩阵 K
$$
K=
\begin{pmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{pmatrix}
$$

## 3.3 外参（相机位姿）
$$
[R|t] =
\begin{pmatrix}
R & t
\end{pmatrix}
$$

## 3.4 归一化坐标
$$
x_n = \frac{X}{Z},\quad y_n = \frac{Y}{Z}
$$

---

# 4. 光照模型（Illumination Model）

## 4.1 Lambertian 反射模型
$$
I = \rho (\mathbf{n} \cdot \mathbf{s})
$$

其中  
- $\rho$：反射率  
- $\mathbf{n}$：法向量  
- $\mathbf{s}$：光源方向  

## 4.2 Phong 模型（更加真实）
$$
I = k_d (\mathbf{n}\cdot\mathbf{l}) + k_s (\mathbf{r}\cdot\mathbf{v})^\alpha
$$

## 4.3 阴影条件
$$
\mathbf{n}\cdot\mathbf{l} < 0\quad \Rightarrow\quad I=0
$$

---

# 5. 卷积与图像滤波（Convolution in Vision）

## 5.1 二维卷积
$$
Y[i,j] = \sum_m\sum_n X[i-m, j-n]\, K[m,n]
$$

## 5.2 卷积的频域解释
$$
X * K \quad \Longleftrightarrow \quad \mathcal{F}(X) \cdot \mathcal{F}(K)
$$

## 5.3 Sobel 边缘算子
$$
G_x=
\begin{pmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{pmatrix}
$$

## 5.4 高斯滤波
$$
G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}
$$

---

# 6. 特征提取与描述（Feature Extraction）

## 6.1 梯度幅值
$$
M = \sqrt{G_x^2 + G_y^2}
$$

方向：
$$
\theta = \arctan\frac{G_y}{G_x}
$$

## 6.2 Harris 角点
矩阵：
$$
M=
\begin{pmatrix}
I_x^2 & I_x I_y\\
I_x I_y & I_y^2
\end{pmatrix}
$$

响应函数：
$$
R = \det(M) - k\,(\text{trace}(M))^2
$$

## 6.3 SIFT 特征
尺度空间：
$$
L(x,y,\sigma)=G(x,y,\sigma) * I(x,y)
$$

DoG：
$$
D=L(x,y,k\sigma)-L(x,y,\sigma)
$$

---

# 7. 深度卷积网络的数学基础（CNN Mathematics）

## 7.1 卷积层
$$
Y_{i,j,c_o} = \sum_{c_i}\sum_{m,n} X_{i+m,j+n,c_i} W_{m,n,c_i,c_o}
$$

## 7.2 ReLU
$$
\text{ReLU}(x)=\max(0,x)
$$

## 7.3 BatchNorm
$$
\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\quad y=\gamma \hat{x}+\beta
$$

## 7.4 池化
最大池化：
$$
Y[i,j]=\max_{(m,n)\in\Omega} X[i+m,j+n]
$$

平均池化：
$$
Y[i,j]=\frac{1}{|\Omega|}\sum_{(m,n)\in\Omega}X[i+m,j+n]
$$

---

# 8. Transformer / ViT 的数学基础

## 8.1 Self-Attention
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

## 8.2 Patch Embedding
图像被切成：
$$
\text{Patch}\in \mathbb{R}^{(P \times P \times C)}
$$

线性投影：
$$
z_p = W_{patch}\, \text{vec}(\text{Patch})
$$

## 8.3 视觉 Transformer 的本质

CV 中自注意力可以理解为：
$$
\text{自相关（信号处理）} \approx QK^T
$$

---

# 9. 光流（Optical Flow）数学基础

## 9.1 稳定假设（Brightness Constancy）
$$
I(x,y,t)=I(x+u, y+v, t+1)
$$

泰勒展开：
$$
I_x u + I_y v + I_t = 0
$$

## 9.2 光流方程
$$
\begin{pmatrix}
I_x\\I_y
\end{pmatrix}
\cdot
\begin{pmatrix}
u\\v
\end{pmatrix}
= -I_t
$$

---

# 10. 优化（Optimization）

## 10.1 梯度下降
$$
\theta_{t+1} = \theta_t - \eta\, \nabla_\theta L
$$

## 10.2 Adam
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

更新：
$$
\theta_{t+1}=\theta_t - \eta\,\frac{m_t}{\sqrt{v_t}+\epsilon}
$$

---

# 11. 反向传播（Backpropagation）

链式法则：
$$
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial x}
$$

卷积的梯度：
$$
\frac{\partial L}{\partial W} = X * \delta
$$
$$
\frac{\partial L}{\partial X} = W_{\text{rot180}} * \delta
$$

---

# 12. 流形（Manifold）与特征空间

CV 中数据通常存在于低维流形上：

$$
\mathcal{M} \subset \mathbb{R}^N
$$

例如：
- 人脸图像在高维像素空间中呈现低维结构  
- 相机位姿形成 $SE(3)$ 流形  

局部线性嵌入（LLE）：
$$
X_i = \sum_j w_{ij} X_j
$$

---

# 13. 总结

本章给出了计算机视觉中最核心的数学基础，包括：

- 张量与多维信号  
- 图像几何与投影  
- 相机模型  
- 光照与物理渲染  
- 卷积/特征提取数学  
- CNN 与 Transformer 的数学结构  
- 优化与 BP  
- 视觉特征空间的流形结构  

适合作为研究者的数学参考基础。
