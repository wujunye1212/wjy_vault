# 信号处理基础（Signal Processing Fundamentals）

## 1. 引言

信号处理研究信号随时间或空间的变化特性，目标包括特征提取、降噪、频域分析与重建。其理论构成了现代深度学习、计算机视觉（CV）、语音分析、脑机接口（BCI）等领域的重要基础。

---

# 2. 离散时间信号（Discrete-Time Signals）

## 2.1 定义

离散时间信号为定义在整数集上的函数：

x[n],n∈Zx[n],\quad n \in \mathbb{Z}x[n],n∈Z

常见示例：

- 图像：x[m,n]x[m,n]x[m,n]
    
- 视频：x[m,n,t]x[m,n,t]x[m,n,t]
    
- EEG：x[n]x[n]x[n]
    
- 语音：x[n]x[n]x[n]
    

## 2.2 基本运算

**移位：**

x[n−n0]x[n-n_0]x[n−n0​]

**缩放：**

a x[n]a\,x[n]ax[n]

**反转：**

x[−n]x[-n]x[−n]

**线性组合（叠加性）：**

ax1[n]+bx2[n]a x_1[n] + b x_2[n]ax1​[n]+bx2​[n]

---

# 3. 离散傅里叶变换（DFT）与 FFT

## 3.1 DFT

X[k]=∑n=0N−1x[n]e−j2πnk/N,k=0,…,N−1X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi nk/N},\quad k=0,\ldots,N-1X[k]=n=0∑N−1​x[n]e−j2πnk/N,k=0,…,N−1

逆变换：

x[n]=1N∑k=0N−1X[k]ej2πnk/Nx[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k]e^{j2\pi nk/N}x[n]=N1​k=0∑N−1​X[k]ej2πnk/N

意义：

- X[k]X[k]X[k]：第 kkk 个频率分量的幅度与相位
    
- 低频：平滑结构
    
- 高频：突变与噪声
    

## 3.2 FFT

FFT 将 DFT 从

O(N2)→O(Nlog⁡N)O(N^2)\quad \rightarrow\quad O(N\log N)O(N2)→O(NlogN)

是快速频域分析与卷积加速的核心方法。

---

# 4. 卷积与滤波（Convolution & Filtering）

## 4.1 一维卷积

(y∗h)[n]=∑k=−∞∞x[k]  h[n−k](y * h)[n] = \sum_{k=-\infty}^{\infty} x[k]\;h[n-k](y∗h)[n]=k=−∞∑∞​x[k]h[n−k]

二维卷积（图像）：

Y[m,n]=∑i∑jX[m−i,n−j]H[i,j]Y[m,n]=\sum_{i}\sum_{j} X[m-i,n-j]H[i,j]Y[m,n]=i∑​j∑​X[m−i,n−j]H[i,j]

## 4.2 卷积定理

x[n]∗h[n]⟺X[k] H[k]x[n] * h[n]\quad \Longleftrightarrow \quad X[k]\,H[k]x[n]∗h[n]⟺X[k]H[k]

含义：

- 时域卷积 = 频域乘法
    
- CNN 卷积本质是 FIR 滤波
    

---

# 5. 滤波器（Filters）

## 5.1 类型

|类型|作用|
|---|---|
|低通|抑制高频噪声|
|高通|强化边缘／快速变化|
|带通|提取某一频段（EEG 常用）|
|带阻|去除某频段（如工频噪声 50/60Hz）|

## 5.2 FIR 与 IIR

### FIR（Finite Impulse Response）

y[n]=∑i=0Mbix[n−i]y[n] = \sum_{i=0}^{M} b_i x[n-i]y[n]=i=0∑M​bi​x[n−i]

- 固有稳定
    
- 支持线性相位（适合 EEG）
    

### IIR（Infinite Impulse Response）

y[n]=∑i=0Mbix[n−i]−∑j=1Kajy[n−j]y[n] = \sum_{i=0}^{M} b_i x[n-i] - \sum_{j=1}^{K} a_j y[n-j]y[n]=i=0∑M​bi​x[n−i]−j=1∑K​aj​y[n−j]

- 有反馈结构
    
- 阶数低、计算快
    

---

# 6. 时频分析（Time-Frequency Analysis）

适用于非平稳信号（如 EEG、语音）。

## 6.1 STFT（Short-Time Fourier Transform）

X(t,f)=∑nx[n] w[n−t] e−j2πfnX(t,f)=\sum_{n} x[n]\,w[n-t]\,e^{-j2\pi fn}X(t,f)=n∑​x[n]w[n−t]e−j2πfn

特征：

- 固定时间窗
    
- 得到频谱图（Spectrogram）
    
- 时间与频率分辨率受限
    

## 6.2 小波变换（Wavelet Transform）

W(a,b)=∫x(t) ψa,b(t) dtW(a,b)=\int x(t)\,\psi_{a,b}(t)\,dtW(a,b)=∫x(t)ψa,b​(t)dt

特点：

- 多分辨率
    
- 高频 → 高时间分辨率
    
- 低频 → 높은频率分辨率
    
- 适合 EEG、脑区振荡分析
    

---

# 7. 随机信号模型（Stochastic Signal Models）

## 7.1 平稳性（WSS）

均值：

E[x[n]]=μ\mathbb{E}[x[n]] = \muE[x[n]]=μ

自相关：

Rx[m]=E[x[n] x[n+m]]R_x[m] = \mathbb{E}[x[n]\,x[n+m]]Rx​[m]=E[x[n]x[n+m]]

功率谱密度（Wiener–Khinchin）：

Sx(ω)=∑m=−∞∞Rx[m]e−jωmS_x(\omega)=\sum_{m=-\infty}^{\infty}R_x[m]e^{-j\omega m}Sx​(ω)=m=−∞∑∞​Rx​[m]e−jωm

## 7.2 AR / ARMA 模型

**AR(p)：**

x[n]=∑k=1pakx[n−k]+ϵ[n]x[n]=\sum_{k=1}^{p}a_k x[n-k] + \epsilon[n]x[n]=k=1∑p​ak​x[n−k]+ϵ[n]

常用于 EEG 频段特征估计（如 Yule-Walker AR 频谱）。

---

# 8. 非平稳信号处理（EEG/BCI）

EEG 特点：非平稳、噪声重、1/f 结构、存在肌电/眼电干扰。

常用操作：

1. 去趋势（Detrending）
    
2. 带通滤波（0.5–40 Hz）
    
3. 工频陷波（50/60Hz Notch）
    
4. ICA 去伪迹（EOG/EMG）
    
5. STFT 或 Wavelet 时频图
    
6. 各频段功率提取（α、β、γ等）
    

---

# 9. 信号处理与深度学习的联系

|信号处理概念|深度学习对应|
|---|---|
|卷积|CNN 卷积核 = FIR 滤波器|
|频域卷积|FFT 卷积，加速大核卷积|
|自相关|注意力机制的 QKTQK^TQKT|
|功率谱|Transformer 可视化分析|
|STFT、小波|EEG、语音特征输入网络|
|滤波器组|多尺度卷积、Scattering Transform|