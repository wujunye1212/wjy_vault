# 信号处理基础（Signal Processing Fundamentals）

## 1. 引言

信号处理研究信号的结构、频谱、随机性与非平稳性，是计算机视觉、语音处理与脑机接口中的基础工具。

---

## 2. 离散时间信号（Discrete-Time Signals）

### 2.1 定义

$x[n],n∈Zx[n], \quad n \in \mathbb{Z}x[n],n∈Z$

常见类型：一维时间信号、二维图像、三维视频、EEG 时间序列。

### 2.2 基本运算

**移位：**

x[n−n0]x[n-n_0]x[n−n0​]$

**缩放：**

a x[n]a\,x[n]ax[n]

**反转：**

x[−n]x[-n]x[−n]

**线性组合：**

ax1[n]+bx2[n]a x_1[n] + b x_2[n]ax1​[n]+bx2​[n]

---

## 3. 离散傅里叶变换（DFT）与 FFT

### 3.1 DFT

X[k]=∑n=0N−1x[n]e−j2πnk/NX[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi nk/N}X[k]=n=0∑N−1​x[n]e−j2πnk/N

逆变换：

x[n]=1N∑k=0N−1X[k]ej2πnk/Nx[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k]e^{j2\pi nk/N}x[n]=N1​k=0∑N−1​X[k]ej2πnk/N

### 3.2 FFT

FFT 将计算复杂度从：

O(N2)→O(Nlog⁡N)O(N^2) \rightarrow O(N\log N)O(N2)→O(NlogN)

---

## 4. 卷积与滤波（Convolution and Filtering）

### 4.1 卷积定义

一维卷积：

(y∗h)[n]=∑k=−∞∞x[k] h[n−k](y * h)[n] = \sum_{k=-\infty}^{\infty} x[k]\,h[n-k](y∗h)[n]=k=−∞∑∞​x[k]h[n−k]

二维卷积（图像）：

Y[m,n]=∑i∑jX[m−i,n−j]H[i,j]Y[m,n]=\sum_{i}\sum_{j} X[m-i,n-j]H[i,j]Y[m,n]=i∑​j∑​X[m−i,n−j]H[i,j]

### 4.2 卷积定理

x[n]∗h[n]⟺X[k] H[k]x[n] * h[n] \Longleftrightarrow X[k]\,H[k]x[n]∗h[n]⟺X[k]H[k]

---

## 5. 滤波器（Filters）

### 5.1 常见类型

- 低通滤波器（LPF）
    
- 高通滤波器（HPF）
    
- 带通滤波器（BPF）
    
- 带阻滤波器（BSF）
    

### 5.2 FIR 与 IIR

**FIR：**

y[n]=∑i=0Mbix[n−i]y[n] = \sum_{i=0}^{M} b_i x[n-i]y[n]=i=0∑M​bi​x[n−i]

**IIR：**

y[n]=∑i=0Mbix[n−i]−∑j=1Kajy[n−j]y[n] = \sum_{i=0}^{M} b_i x[n-i] - \sum_{j=1}^{K} a_j y[n-j]y[n]=i=0∑M​bi​x[n−i]−j=1∑K​aj​y[n−j]

---

## 6. 时频分析（Time-Frequency Analysis）

### 6.1 STFT（Short-Time Fourier Transform）

X(t,f)=∑nx[n] w[n−t] e−j2πfnX(t,f)=\sum_{n} x[n]\,w[n-t]\,e^{-j2\pi fn}X(t,f)=n∑​x[n]w[n−t]e−j2πfn

得到时间–频率谱（Spectrogram）。

### 6.2 小波变换（Wavelet Transform）

W(a,b)=∫x(t) ψa,b(t) dtW(a,b)=\int x(t)\,\psi_{a,b}(t)\,dtW(a,b)=∫x(t)ψa,b​(t)dt

具有多分辨率特性。

---

## 7. 随机信号模型（Stochastic Signal Models）

### 7.1 平稳性（WSS）

均值：

E[x[n]]=μ\mathbb{E}[x[n]]=\muE[x[n]]=μ

自相关：

Rx[m]=E[x[n]x[n+m]]R_x[m]=\mathbb{E}[x[n]x[n+m]]Rx​[m]=E[x[n]x[n+m]]

功率谱密度（PSD）：

Sx(ω)=∑m=−∞∞Rx[m]e−jωmS_x(\omega)=\sum_{m=-\infty}^{\infty}R_x[m]e^{-j\omega m}Sx​(ω)=m=−∞∑∞​Rx​[m]e−jωm

### 7.2 AR 模型

x[n]=∑k=1pakx[n−k]+ϵ[n]x[n]=\sum_{k=1}^{p} a_k x[n-k]+\epsilon[n]x[n]=k=1∑p​ak​x[n−k]+ϵ[n]

---

## 8. 非平稳信号处理（EEG/BCI）

常用处理步骤：

1. 去趋势
    
2. 带通滤波（0.5–40 Hz）
    
3. 工频陷波（50/60 Hz）
    
4. ICA 去伪迹
    
5. STFT 或小波变换
    
6. 各频段功率提取
    

---

## 9. 信号处理与深度学习的对应关系

|信号处理概念|深度学习对应|
|---|---|
|卷积|CNN 卷积核|
|频域卷积|FFT 卷积|
|自相关|注意力中的 QKTQK^TQKT|
|功率谱|Transformer 特征可视化|
|STFT/小波|时频特征输入模型|

---

## 10. 总结

信号处理包括离散信号表示、傅里叶变换、卷积滤波、时频分析、随机过程建模以及 EEG 的非平稳处理，是视觉、语音、EEG 与深度模型设计的共同数学基础。