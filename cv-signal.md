# 信号处理基础（Complete Signal Processing Fundamentals）

---

# 1. 信号与系统概述

## 1.1 信号定义
信号是携带信息的物理量，可表示为：
$$
x(t),\; x[n]
$$
其中 $t$ 表连续时间，$n$ 表离散时间。

## 1.2 信号类型
- **连续/离散**  
- **模拟/数字**  
- **周期/非周期**  
- **确定/随机**  
- **一维（音频）、二维（图像）、多维（视频）**

---

# 2. 基本信号及性质

## 2.1 单位冲激
$$
\delta[n] =
\begin{cases}
1, & n=0 \\
0, & n\neq 0
\end{cases}
$$

## 2.2 单位阶跃
$$
u[n]=
\begin{cases}
1, & n\ge 0\\
0, & n < 0
\end{cases}
$$

## 2.3 复指数
$$
x[n] = e^{j\omega n}
$$

---

# 3. 离散时间系统

## 3.1 线性时不变系统（LTI）
满足：
- 线性：$ax_1[n]+bx_2[n]$ → $a y_1[n] + b y_2[n]$
- 时不变：$x[n-n_0]$ → $y[n-n_0]$

## 3.2 冲激响应与卷积
LTI 系统完全由 $h[n]$ 决定：

$$
y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k] h[n-k]
$$

---

# 4. Z 变换（Z-Transform）

## 4.1 定义
$$
X(z) = \sum_{n=-\infty}^{\infty} x[n] z^{-n}
$$

## 4.2 收敛域（ROC）
决定信号是否稳定：
- 包含单位圆 → 稳定系统  
- 不包含单位圆 → 非稳定

## 4.3 逆变换
部分分式展开求解。

---

# 5. 离散傅里叶变换（DFT）

## 5.1 DFT 定义
$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi nk/N}
$$

## 5.2 DFT 性质
线性：
$$
a x[n] + b y[n] \rightarrow a X[k] + b Y[k]
$$

循环卷积：
$$
x[n] \circledast h[n] \Longleftrightarrow X[k] H[k]
$$

移位：
$$
x[n-n_0] \rightarrow X[k] e^{-j2\pi k n_0/N}
$$

---

# 6. 快速傅里叶变换（FFT）

FFT 用分治法将 $O(N^2)$ 计算降低至：
$$
O(N\log N)
$$

用途：
- 高频滤波
- 频谱估计
- CNN 中的 FFT 卷积加速

---

# 7. 数字滤波器设计

## 7.1 FIR 滤波器
有限冲激响应：
$$
y[n] = \sum_{i=0}^M b_i x[n-i]
$$

特点：
- 永久稳定
- 线性相位（适合 EEG）

常见设计：
- 窗函数法（Hamming、Blackman）
- 最小二乘法（LS）
- Parks–McClellan（Remez）

## 7.2 IIR 滤波器
无限冲激响应：
$$
y[n] = \sum_{i=0}^M b_i x[n-i] - \sum_{j=1}^K a_j y[n-j]
$$

经典原型：
- 巴特沃斯（平坦幅度）
- 切比雪夫（陡峭通带）
- 椭圆（最陡峭）

---

# 8. 随机信号与谱分析

## 8.1 均值
$$
\mu_x = \mathbb{E}[x[n]]
$$

## 8.2 自相关函数
$$
R_x[m] = \mathbb{E}[x[n]x[n+m]]
$$

## 8.3 功率谱密度（PSD）
$$
S_x(\omega) = \sum_{m=-\infty}^{\infty} R_x[m] e^{-j\omega m}
$$

## 8.4 常用模型：AR、MA、ARMA

AR(p)：
$$
x[n] = \sum_{k=1}^p a_k x[n-k] + \epsilon[n]
$$

AR 常用于 EEG 频谱估计（如 Yule–Walker 方法）。

---

# 9. 时频分析（Time-Frequency）

## 9.1 STFT（短时傅里叶变换）
$$
X(t,f) = \sum_n x[n] w[n-t] e^{-j 2\pi f n}
$$
得到谱图（Spectrogram）。

## 9.2 CWT（连续小波变换）
$$
W(a,b) = \int x(t)\, \psi_{a,b}(t)\, dt
$$

特点：
- 高频 → 高时间分辨率  
- 低频 → 高频率分辨率  

## 9.3 多分辨率分析（MRA）
构成小波基的理论基础。

---

# 10. 高级频谱分析

## 10.1 Welch 法（降低方差）

步骤：
1. 分段 → 加窗  
2. FFT  
3. 平均  

伪公式：
$$
S(\omega)=\frac{1}{K}\sum_{i=1}^K |FFT(x_i)|^2
$$

## 10.2 多锥波（Multitaper）
使用多个正交 tapers 减少谱泄漏：

$$
S(\omega)=\frac{1}{K}\sum_{i=1}^K |FFT(x[n] h_i[n])|^2
$$

---

# 11. 信号采样理论（Sampling Theory）

## 11.1 奈奎斯特频率
$$
f_s \ge 2 f_{\max}
$$

## 11.2 混叠（Aliasing）
当采样率不足，频率折叠到低频部分。

分析：
$$
X_{\text{sampled}}(\omega)=\frac{1}{T} \sum_k X(\omega - k\omega_s)
$$

## 11.3 反混叠滤波
采样前必须进行低通滤波。

---

# 12. 滤波器组（Filter Banks）

## 12.1 匹配滤波（Matched Filter）
用于最大化信噪比：

$$
h[n] = x[N-n]
$$

## 12.2 Mel 滤波器组（语音）
形成 MFCC 特征：
$$
E_m = \sum |X[k]|^2 H_m[k]
$$

## 12.3 EEG 频带滤波（BCI）
- Delta：0.5–4 Hz  
- Theta：4–8 Hz  
- Alpha：8–13 Hz  
- Beta：13–30 Hz  
- Gamma：30–80 Hz  

---

# 13. 多通道信号与空间滤波（EEG/BCI）

## 13.1 协方差矩阵
$$
C = \frac{1}{T} XX^T
$$

## 13.2 空间谱
均值场：
$$
P(\omega) = w^T C(\omega) w
$$

## 13.3 CSP（共空间模式）
最大化两个类别的方差比：

$$
W = \arg\max_w \frac{w^T C_1 w}{w^T C_2 w}
$$

核心工具用于 BCI 二分类。

---

# 14. 时频–图结构（现代 EEG 深度模型）

现代深度模型常把信号转为图结构：

- 节点：通道  
- 边：相关系数、相位同步、距离  

例如 Laplacian：
$$
L = D - A
$$

Chebyshev GCN 中使用：
$$
g_\theta(L)x \approx \sum_{k=0}^K \theta_k T_k(\tilde{L}) x
$$

Transformer 注意力与信号自相关关系：
$$
\text{Attention}(Q,K,V) \approx R_x \cdot V
$$

---

# 15. 总结

本教材级内容涵盖：

- 信号与系统  
- Z 变换  
- DFT/FFT  
- FIR/IIR 设计  
- 随机信号与谱估计  
- 时频分析：STFT、CWT、Welch、多锥波  
- 采样理论  
- 多通道 EEG 与空间滤波  
- 信号处理与现代深度学习（GCN/Transformer）的映射  

适合科研、工程和教学使用。
