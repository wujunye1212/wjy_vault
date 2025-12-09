# 信号处理基础（Signal Processing Fundamentals）

## 1. 引言
信号处理研究信号的结构、频率特性、随机性与非平稳性，是 CV、语音分析与脑机接口的基础。

---

## 2. 离散时间信号（Discrete-Time Signals）

### 2.1 定义

$$
x[n],\quad n\in\mathbb{Z}
$$

### 2.2 基本运算

移位：
$$
x[n-n_0]
$$

缩放：
$$
a\,x[n]
$$

反转：
$$
x[-n]
$$

线性组合：
$$
a x_1[n] + b x_2[n]
$$

---

## 3. 离散傅里叶变换（DFT）与 FFT

### 3.1 DFT

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi nk/N}
$$

逆变换：
$$
x[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k] e^{j2\pi nk/N}
$$

### 3.2 FFT

$$
O(N^2)\rightarrow O(N\log N)
$$

---

## 4. 卷积与滤波

### 4.1 一维卷积
$$
(y*h)[n]=\sum_{k=-\infty}^{\infty}x[k]\,h[n-k]
$$

二维卷积：
$$
Y[m,n]=\sum_i\sum_j X[m-i,n-j]\,H[i,j]
$$

### 4.2 卷积定理
$$
x[n]*h[n] \Longleftrightarrow X[k]\,H[k]
$$

---

## 5. 滤波器（Filters）

### 5.1 类型
- 低通（LPF）
- 高通（HPF）
- 带通（BPF）
- 带阻（BSF）

### 5.2 FIR / IIR

FIR:
$$
y[n] = \sum_{i=0}^M b_i x[n-i]
$$

IIR:
$$
y[n] = \sum_{i=0}^M b_i x[n-i] - \sum_{j=1}^K a_j y[n-j]
$$

---

## 6. 时频分析（Time-Frequency Analysis）

### 6.1 STFT
$$
X(t,f)=\sum_n x[n]\,w[n-t]\,e^{-j2\pi fn}
$$

### 6.2 小波变换
$$
W(a,b)=\int x(t)\,\psi_{a,b}(t)\,dt
$$

---

## 7. 随机信号模型（Stochastic Signal Models）

### 7.1 宽平稳过程（WSS）

均值：
$$
\mathbb{E}[x[n]]=\mu
$$

自相关：
$$
R_x[m]=\mathbb{E}[x[n]\,x[n+m]]
$$

功率谱密度（PSD）：
$$
S_x(\omega)=\sum_{m=-\infty}^{\infty} R_x[m] e^{-j\omega m}
$$

### 7.2 AR 模型
$$
x[n]=\sum_{k=1}^p a_k x[n-k] + \epsilon[n]
$$

---

## 8. 非平稳信号处理（EEG/BCI）

1. 去趋势  
2. 0.5–40 Hz 带通  
3. 工频陷波（50/60 Hz）  
4. ICA 去伪迹  
5. STFT/小波  
6. 频段功率（$$\alpha,\beta,\gamma$$）

---

## 9. 信号处理与深度学习对应

| 信号处理概念 | 深度学习映射 |
|--------------|--------------|
| 卷积 | CNN 卷积核 |
| 自相关 | 注意力中的 $$QK^T$$ |
| STFT/小波 | 时频特征输入 |
| 频域卷积 | FFT 卷积 |

---

## 10. 总结
信号处理构成 CNN、Transformer、GNN 以及 BCI 中 EEG 分析的数学基础。
