# 信号处理基础（Signal Processing Fundamentals）

## 1. 引言

信号处理研究信号的结构、频率特性、随机性与非平稳性，是 CV、语音分析与脑机接口的基础。

---

## 2. 离散时间信号（Discrete-Time Signals）

### 2.1 定义

x[n],n∈Zx[n],\quad n\in\mathbb{Z}x[n],n∈Z

### 2.2 基本运算

**移位：**

x[n−n0]x[n-n_0]x[n−n0​]

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

x[n]=1N∑k=0N−1X[k]ej2πnk/Nx[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k] e^{j2\pi nk/N}x[n]=N1​k=0∑N−1​X[k]ej2πnk/N

### 3.2 FFT

计算复杂度变化：

O(N2)→O(Nlog⁡N)O(N^2)\rightarrow O(N\log N)O(N2)→O(NlogN)

---

## 4. 卷积与滤波

### 4.1 卷积定义

一维卷积：

(y∗h)[n]=∑k=−∞∞x[k] h[n−k](y*h)[n]=\sum_{k=-\infty}^{\infty}x[k]\,h[n-k](y∗h)[n]=k=−∞∑∞​x[k]h[n−k]

二维卷积：

Y[m,n]=∑i∑jX[m−i,n−j] H[i,j]Y[m,n]=\sum_i\sum_j X[m-i,n-j]\,H[i,j]Y[m,n]=i∑​j∑​X[m−i,n−j]H[i,j]

### 4.2 卷积定理

x[n]∗h[n]⟺X[k] H[k]x[n]*h[n] \Longleftrightarrow X[k]\,H[k]x[n]∗h[n]⟺X[k]H[k]

---

## 5. 滤波器（Filters）

### 5.1 类型

- 低通（LPF）
    
- 高通（HPF）
    
- 带通（BPF）
    
- 带阻（BSF）
    

### 5.2 FIR / IIR

**FIR：**

y[n]=∑i=0Mbix[n−i]y[n] = \sum_{i=0}^M b_i x[n-i]y[n]=i=0∑M​bi​x[n−i]

**IIR：**

y[n]=∑i=0Mbix[n−i]−∑j=1Kajy[n−j]y[n] = \sum_{i=0}^M b_i x[n-i] - \sum_{j=1}^K a_j y[n-j]y[n]=i=0∑M​bi​x[n−i]−j=1∑K​aj​y[n−j]

---

## 6. 时频分析（Time-Frequency Analysis）

### 6.1 STFT

X(t,f)=∑nx[n] w[n−t] e−j2πfnX(t,f)=\sum_n x[n]\,w[n-t]\,e^{-j2\pi fn}X(t,f)=n∑​x[n]w[n−t]e−j2πfn

### 6.2 小波变换（Wavelet Transform）

W(a,b)=∫x(t) ψa,b(t) dtW(a,b)=\int x(t)\,\psi_{a,b}(t)\,dtW(a,b)=∫x(t)ψa,b​(t)dt

---

## 7. 随机信号模型（Stochastic Signal Models）

### 7.1 宽平稳过程（WSS）

均值：

E[x[n]]=μ\mathbb{E}[x[n]]=\muE[x[n]]=μ

自相关：

Rx[m]=E[x[n] x[n+m]]R_x[m]=\mathbb{E}[x[n]\,x[n+m]]Rx​[m]=E[x[n]x[n+m]]

功率谱密度（PSD）：

Sx(ω)=∑m=−∞∞Rx[m]e−jωmS_x(\omega)=\sum_{m=-\infty}^{\infty} R_x[m] e^{-j\omega m}Sx​(ω)=m=−∞∑∞​Rx​[m]e−jωm

### 7.2 AR 模型

x[n]=∑k=1pakx[n−k]+ϵ[n]x[n]=\sum_{k=1}^p a_k x[n-k] + \epsilon[n]x[n]=k=1∑p​ak​x[n−k]+ϵ[n]

---

## 8. 非平稳信号处理（EEG/BCI）

主要步骤：

1. 去趋势
    
2. 0.5–40 Hz 带通
    
3. 50/60 Hz 陷波
    
4. ICA 去伪迹
    
5. STFT/小波
    
6. 各频段功率（$\alpha,\beta,\gamma$ 等）
    

---

## 9. 信号处理与深度学习的映射

|信号处理概念|深度学习对应|
|---|---|
|卷积|CNN 卷积核|
|自相关|注意力中的 $QK^T$|
|STFT/小波|时频特征|
|频域卷积|FFT 卷积|

---

## 10. 总结

信号处理构成深度学习模型设计（CNN、Transformer、GNN）与 BCI 信号分析的数学基础，包括傅里叶分析、卷积滤波、时频变换与随机过程建模。