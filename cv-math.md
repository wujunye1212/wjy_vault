# 计算机视觉完整知识体系

## 第一部分：基础理论(Fundamental Theory)

### 一、数学基础(Mathematical Foundations)

#### 1.1 线性代数与矩阵理论

**1.1.1 向量空间与线性变换**

向量空间$V$定义为满足加法与标量乘法封闭性的集合。若存在线性无关的向量集合${\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n}$张成$V$，则称其为$V$的一组基，$n$为维数$\dim(V)$。

线性变换$T: V \rightarrow W$满足： $$T(\alpha \mathbf{u} + \beta \mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v}), \quad \forall \alpha, \beta \in \mathbb{F}, \mathbf{u}, \mathbf{v} \in V$$

对应的矩阵表示$A \in \mathbb{R}^{m \times n}$，其中$m = \dim(W)$，$n = \dim(V)$。

**秩与零空间**

秩(Rank)：矩阵$A$列向量线性无关的最大数目 $$\text{rank}(A) = \dim(\text{Im}(A)) = \dim(\text{Col}(A))$$

零空间(Null Space)： $$\text{Null}(A) = {\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{0}}$$

秩-零定理(Rank-Nullity Theorem)： $$\text{rank}(A) + \text{nullity}(A) = n$$

其中$\text{nullity}(A) = \dim(\text{Null}(A))$

---

**1.1.2 矩阵分解理论**

**特征值分解(Eigenvalue Decomposition, EVD)**

对于$n \times n$方阵$A$，若存在标量$\lambda$和非零向量$\mathbf{v}$满足： $$A\mathbf{v} = \lambda\mathbf{v}$$

则$\lambda$为特征值(Eigenvalue)，$\mathbf{v}$为对应的特征向量(Eigenvector)。

若$A$为对称矩阵($A = A^T$)，则存在正交矩阵$Q$使得： $$A = Q\Lambda Q^{-1} = Q\Lambda Q^T$$

其中$Q$的列向量为标准正交特征向量，$\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$为特征值对角矩阵。

**重要性质**：

- 迹(Trace)：$\text{tr}(A) = \sum_{i=1}^n \lambda_i$
- 行列式(Determinant)：$\det(A) = \prod_{i=1}^n \lambda_i$

---

**奇异值分解(Singular Value Decomposition, SVD)**

对于任意$m \times n$矩阵$A$，存在正交矩阵$U \in \mathbb{R}^{m \times m}$、$V \in \mathbb{R}^{n \times n}$和对角矩阵$\Sigma \in \mathbb{R}^{m \times n}$使得： $$A = U\Sigma V^T$$

其中$\Sigma = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_r, 0, \ldots, 0)$，$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$为奇异值(Singular Values)，$r = \text{rank}(A)$。

**几何解释**：

- $V$的列向量：$A$的列空间方向
- $U$的列向量：$A$作用后的方向
- $\Sigma$中的元素：各方向上的缩放因子

**应用**：

1. 秩估计：$\text{rank}(A) = {\sigma_i > 0}$
    
2. 伪逆(Moore-Penrose Pseudoinverse)： $$A^+ = V\Sigma^+ U^T$$ 其中$\Sigma^+$将$\Sigma$中非零元素取倒数，零保持为零。
    
3. 最小二乘解： $$\mathbf{x}^* = A^+\mathbf{b} = \arg\min_\mathbf{x} |A\mathbf{x} - \mathbf{b}|_2^2$$
    

---

**QR分解**

对于$m \times n$矩阵$A$（$m \geq n$），存在正交矩阵$Q \in \mathbb{R}^{m \times n}$和上三角矩阵$R \in \mathbb{R}^{n \times n}$使得： $$A = QR$$

其中$Q^T Q = I_n$，$R$为上三角矩阵。

**数值优势**：

- 条件数改善：$\kappa(A) \rightarrow \kappa(R)$，$\kappa(R) \leq \kappa(A)$
- 避免正规方程$A^T A$的平方效应

---

**Cholesky分解**

对于对称正定矩阵$A \in \mathbb{R}^{n \times n}$，存在唯一下三角矩阵$L$使得： $$A = LL^T$$

其中$L$的对角元素为正。

**应用**：

- 高效求解线性系统：$A\mathbf{x} = \mathbf{b}$ $\Rightarrow$ 两次三角求解
- 协方差矩阵分解
- 行列式计算：$\det(A) = (\det(L))^2 = (\prod_{i=1}^n L_{ii})^2$

---

**1.1.3 矩阵范数与条件数**

**向量范数**

$\mathbb{R}^n$上的范数$|\cdot|$满足：

1. 正定性：$|\mathbf{x}| \geq 0$，等号成立当且仅当$\mathbf{x} = \mathbf{0}$
2. 齐次性：$|\alpha\mathbf{x}| = |\alpha||\mathbf{x}|$
3. 三角不等式：$|\mathbf{x} + \mathbf{y}| \leq |\mathbf{x}| + |\mathbf{y}|$

常用范数： $$|\mathbf{x}|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}, \quad p \in [1, \infty)$$

特殊情形：

- $L_1$范数(曼哈顿距离)：$|\mathbf{x}|_1 = \sum_{i=1}^n |x_i|$
- $L_2$范数(欧氏距离)：$|\mathbf{x}|_2 = \sqrt{\sum_{i=1}^n x_i^2}$
- $L_\infty$范数：$|\mathbf{x}|_\infty = \max_i |x_i|$

---

**矩阵范数**

诱导范数(Induced Norm)定义为： $$|A|_p = \max_{\mathbf{x} \neq \mathbf{0}} \frac{|A\mathbf{x}|_p}{|\mathbf{x}|_p}$$

常用矩阵范数：

1. **谱范数(Spectral Norm)**： $$|A|_2 = \max_i \sigma_i(A) = \sigma_1(A)$$ 其中$\sigma_i$为奇异值。
    
2. **Frobenius范数**： $$|A|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n a_{ij}^2} = \sqrt{\sum_{i=1}^r \sigma_i^2} = \sqrt{\text{tr}(A^T A)}$$
    
3. **核范数(Nuclear Norm)**： $$|A|_* = \sum_{i=1}^r \sigma_i$$
    

**重要性质**：

- $|A\mathbf{x}|_2 \leq |A|_2 |\mathbf{x}|_2$
- $|AB|_F \leq |A|_2 |B|_F$
- 相容性：$|A\mathbf{x}|_p \leq |A|_p |\mathbf{x}|_p$

---

**条件数(Condition Number)**

条件数衡量矩阵对扰动的敏感性： $$\kappa(A) = |A| \cdot |A^{-1}|$$

对于$L_2$范数： $$\kappa_2(A) = \frac{\sigma_1(A)}{\sigma_n(A)}$$

**条件数的含义**：

- $\kappa(A) \approx 1$：矩阵良态，数值稳定
- $\kappa(A) \gg 1$：矩阵病态，易产生数值误差

**误差分析**：

对于线性系统$A\mathbf{x} = \mathbf{b}$，若$\mathbf{b}$有扰动$\delta\mathbf{b}$，则解的相对误差界为： $$\frac{|\delta\mathbf{x}|}{|\mathbf{x}|} \leq \kappa(A) \frac{|\delta\mathbf{b}|}{|\mathbf{b}|}$$

---

#### 1.2 微积分与优化理论

**1.2.1 多元微积分基础**

**梯度(Gradient)**

对于标量函数$f: \mathbb{R}^n \rightarrow \mathbb{R}$，梯度定义为： $$\nabla f(\mathbf{x}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \ \frac{\partial f}{\partial x_2} \ \vdots \ \frac{\partial f}{\partial x_n} \end{pmatrix} \in \mathbb{R}^n$$

梯度指向函数增长最快的方向，其大小为该方向的增长率。

**方向导数(Directional Derivative)**

沿单位向量$\mathbf{d}$的方向导数为： $$\nabla_\mathbf{d} f(\mathbf{x}) = \nabla f(\mathbf{x})^T \mathbf{d}$$

---

**海塞矩阵(Hessian Matrix)**

对于标量函数$f: \mathbb{R}^n \rightarrow \mathbb{R}$，海塞矩阵为二阶偏导数矩阵： $$H(f)(\mathbf{x}) = \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \ \vdots & \vdots & \ddots & \vdots \ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{pmatrix}$$

若$f$二阶连续可微，则$H$为对称矩阵。

**应用于最优性条件**：

- 若$\nabla f(\mathbf{x}^_) = \mathbf{0}$且$H(\mathbf{x}^_) \succ 0$（正定），则$\mathbf{x}^*$为严格局部最小值
- 若$H(\mathbf{x}^_) \prec 0$（负定），则$\mathbf{x}^_$为严格局部最大值

---

**雅可比矩阵(Jacobian Matrix)**

对于向量函数$\mathbf{f}: \mathbb{R}^n \rightarrow \mathbb{R}^m$，$\mathbf{f} = (f_1, f_2, \ldots, f_m)^T$，雅可比矩阵为： $$J(\mathbf{f})(\mathbf{x}) = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \ \vdots & \vdots & \ddots & \vdots \ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix} \in \mathbb{R}^{m \times n}$$

**在计算机视觉中的应用**：非线性优化(如Bundle Adjustment)中计算参数的偏导数。

---

**泰勒展开(Taylor Expansion)**

对于光滑函数$f: \mathbb{R}^n \rightarrow \mathbb{R}$，在$\mathbf{x}_0$处的二阶泰勒展开为： $$f(\mathbf{x}_0 + \Delta\mathbf{x}) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^T H(\mathbf{x}_0) \Delta\mathbf{x} + O(|\Delta\mathbf{x}|^3)$$

其中忽略高阶项后得二阶近似： $$f(\mathbf{x}_0 + \Delta\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^T H(\mathbf{x}_0) \Delta\mathbf{x}$$

---

**1.2.2 凸优化理论**

**凸集(Convex Set)**

集合$C$为凸集，若对任意$\mathbf{x}, \mathbf{y} \in C$和$\theta \in [0,1]$，都有： $$\theta\mathbf{x} + (1-\theta)\mathbf{y} \in C$$

**凸函数(Convex Function)**

函数$f: \mathbb{R}^n \rightarrow \mathbb{R}$为凸函数，若对任意$\mathbf{x}, \mathbf{y} \in \text{dom}(f)$和$\theta \in [0,1]$，都有： $$f(\theta\mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y})$$

**等价条件**（一阶充要条件）：

若$f$可微，则$f$为凸函数当且仅当对任意$\mathbf{x}, \mathbf{y}$： $$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x})$$

**二阶充分条件**：

若$f$二阶可微且$H(f)(\mathbf{x}) \succeq 0$（半正定）对所有$\mathbf{x}$成立，则$f$为凸函数。

**重要性质**：

- 凸函数的任何局部最小值都是全局最小值
- 凸函数的海塞矩阵半正定

---

**Lipschitz连续性(Lipschitz Continuity)**

函数$f: \mathbb{R}^n \rightarrow \mathbb{R}$满足$L$-Lipschitz连续，若存在常数$L \geq 0$使得： $$|f(\mathbf{x}) - f(\mathbf{y})| \leq L|\mathbf{x} - \mathbf{y}|$$

对所有$\mathbf{x}, \mathbf{y}$成立。

**性质**：

- 若$|\nabla f(\mathbf{x})| \leq L$对所有$\mathbf{x}$成立，则$f$满足$L$-Lipschitz连续
- Lipschitz连续的函数通常具有更好的数值稳定性

---

**强凸性(Strong Convexity)**

函数$f: \mathbb{R}^n \rightarrow \mathbb{R}$满足$\mu$-强凸性，若对任意$\mathbf{x}, \mathbf{y}$和$\theta \in [0,1]$： $$f(\theta\mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y}) - \frac{\mu}{2}\theta(1-\theta)|\mathbf{x} - \mathbf{y}|^2$$

**等价条件**（一阶）： $$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x}) + \frac{\mu}{2}|\mathbf{y} - \mathbf{x}|^2$$

**二阶条件**：

若$H(f)(\mathbf{x}) \succeq \mu I$对所有$\mathbf{x}$成立，则$f$满足$\mu$-强凸性。

**优势**：强凸函数有唯一的全局最小值，优化收敛更快。

---

**1.2.3 一阶优化算法**

**梯度下降法(Gradient Descent, GD)**

更新规则： $$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$$

其中$\eta > 0$为学习率(Learning Rate)。

**收敛分析**：

对于$L$-光滑的凸函数，若学习率$\eta \leq \frac{1}{L}$，则GD满足： $$f(\mathbf{x}_t) - f(\mathbf{x}^*) = O\left(\frac{1}{t}\right)$$

即$t$步后的函数值与最优值的差距为$O(1/t)$。

**问题**：

- 学习率难以选择
- 在病态问题上收敛缓慢(对数线性或更差)

---

**随机梯度下降法(Stochastic Gradient Descent, SGD)**

每步仅使用一个或小批量样本的梯度： $$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f_{i_t}(\mathbf{x}_t)$$

其中$i_t$随机抽样。

**优势**：

- 计算效率高
- 对大规模数据集可行
- 随机性有助于逃离局部最小值

**劣势**：

- 噪声导致收敛波动
- 需要学习率衰减策略

---

**动量法(Momentum)**

维护速度向量$\mathbf{v}_t$： $$\mathbf{v}_{t+1} = \beta\mathbf{v}_t + (1-\beta)\nabla f(\mathbf{x}_t)$$ $$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \mathbf{v}_{t+1}$$

其中$\beta \in [0,1)$为动量系数(通常0.9)。

**解释**：

- 累积过去梯度的指数加权平均
- 在梯度方向一致时加速
- 在梯度方向改变时减速

**收敛速度**：在凸强凸问题上快于标准GD。

---

**Adam优化(Adaptive Moment Estimation)**

维护一阶矩$\mathbf{m}_t$和二阶矩$\mathbf{v}_t$： $$\mathbf{m}_{t+1} = \beta_1 \mathbf{m}_t + (1-\beta_1)\nabla f(\mathbf{x}_t)$$ $$\mathbf{v}_{t+1} = \beta_2 \mathbf{v}_t + (1-\beta_2)(\nabla f(\mathbf{x}_t))^{\odot 2}$$

其中$(\cdot)^{\odot 2}$表示逐元素平方。

偏差修正： $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_{t+1}}{1-\beta_1^{t+1}}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_{t+1}}{1-\beta_2^{t+1}}$$

更新： $$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

**参数**：$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$

**优势**：

- 自适应学习率
- 对超参数不敏感
- 在深度学习中表现优异

---

**1.2.4 二阶优化算法**

**牛顿法(Newton's Method)**

利用二阶信息的更新： $$\mathbf{x}_{t+1} = \mathbf{x}_t - H(\mathbf{x}_t)^{-1}\nabla f(\mathbf{x}_t)$$

其中$H(\mathbf{x}_t)$为海塞矩阵。

**收敛性质**：

- 二阶收敛：$\| \mathbf{x}_{t+1} - \mathbf{x}^* \| \leq C \| \mathbf{x}_t - \mathbf{x}^* \|^2$
- 极少次数的迭代到达精度要求

**缺点**：

- 每步需计算和求逆海塞矩阵，计算代价大$O(n^3)$
- 海塞矩阵可能不可逆
- 需要较好的初始点

---

**拟牛顿法(Quasi-Newton Methods)**

用秩2更新来近似海塞矩阵或其逆，避免显式计算。

**BFGS方法**

海塞矩阵的秩2更新： $$H_{t+1} = H_t + \frac{\mathbf{y}_t\mathbf{y}_t^T}{\mathbf{y}_t^T\mathbf{s}_t} - \frac{H_t\mathbf{s}_t\mathbf{s}_t^T H_t}{\mathbf{s}_t^T H_t \mathbf{s}_t}$$

其中$\mathbf{s}_t = \mathbf{x}_{t+1} - \mathbf{x}_t$，$\mathbf{y}_t = \nabla f(\mathbf{x}_{t+1}) - \nabla f(\mathbf{x}_t)$。

**L-BFGS方法**

仅保存最近$m$步的向量对${(\mathbf{s}_i, \mathbf{y}_i)}$，空间复杂度为$O(mn)$而非$O(n^2)$。

**优势**：

- 无需显式计算海塞矩阵
- 收敛速度介于梯度下降和牛顿法之间
- 对大规模问题可行

---

**1.2.5 约束优化**

**拉格朗日乘数法(Lagrange Multiplier Method)**

等式约束问题： $$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) = 0, \quad i = 1, \ldots, m$$

拉格朗日函数： $$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x})$$

必要条件(一阶)： $$∇x​L(x^∗,λ^∗)=0$$ 
  $$ \nabla_{\lambda} \mathcal{L}(\mathbf{x}^*, \lambda^*) = \mathbf{0} $$

---

**KKT条件(Karush-Kuhn-Tucker Conditions)**

不等式与等式混合约束问题： $$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \quad h_j(\mathbf{x}) = 0$$

KKT必要条件：

1. 可行性：$g_i(\mathbf{x}^*) \leq 0,\ h_j(\mathbf{x}^*) = 0$
2. 梯度条件：$\nabla f(\mathbf{x}^*) + \sum_i \mu_i^* \nabla g_i(\mathbf{x}^*) + \sum_j \lambda_j^* \nabla h_j(\mathbf{x}^*) = \mathbf{0}$
3. 互补松弛(Complementary Slackness)：$\mu_i^* g_i(\mathbf{x}^*) = 0,\ \mu_i^* \geq 0$

对于凸问题，KKT条件是充要条件。

---

#### 1.3 概率论与贝叶斯推断

**1.3.1 概率基础**

**概率公理(Probability Axioms)**

对于概率空间$(\Omega, \mathcal{F}, P)$：

1. 非负性：$P(A) \geq 0$，$\forall A \in \mathcal{F}$
2. 规范性：$P(\Omega) = 1$
3. 可列可加性：若$A_i$两两互斥，则$P(\bigcup_i A_i) = \sum_i P(A_i)$

**条件概率(Conditional Probability)**

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**全概率公式(Law of Total Probability)**

若${B_1, B_2, \ldots, B_n}$构成$\Omega$的分割，则： $$P(A) = \sum_{i=1}^n P(A|B_i)P(B_i)$$

---

**贝叶斯定理(Bayes' Theorem)**

$$P(\theta|X) = \frac{P(X|\theta)P(\theta)}{P(X)} = \frac{P(X|\theta)P(\theta)}{\int P(X|\theta')P(\theta')d\theta'}$$

其中：

- $P(\theta|X)$：后验分布(Posterior)
- $P(X|\theta)$：似然函数(Likelihood)
- $P(\theta)$：先验分布(Prior)
- $P(X)$：证据(Evidence)

**在CV中的应用**：根据观测数据更新对场景或模型参数的估计。

---

**1.3.2 常见分布**

**高斯分布(Gaussian Distribution)**

一维高斯分布的概率密度函数(PDF)： $p(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$

其中$\mu$为均值，$\sigma^2$为方差。

多维高斯分布： $p(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{n/2}|\boldsymbol{\Sigma}|^{1/2}}\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$

其中$\boldsymbol{\mu} \in \mathbb{R}^n$为均值向量，$\boldsymbol{\Sigma} \in \mathbb{R}^{n \times n}$为协方差矩阵(对称正定)，$|\boldsymbol{\Sigma}|$为行列式。

**性质**：

- 高斯分布在求解线性问题时导出显式解
- 由线性变换保持高斯性：若$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，则$A\mathbf{x} + \mathbf{b} \sim \mathcal{N}(A\boldsymbol{\mu} + \mathbf{b}, A\boldsymbol{\Sigma}A^T)$

---

**Student-t分布**

概率密度函数： $p(x|\mu, \sigma^2, \nu) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\sigma\Gamma(\frac{\nu}{2})}\left(1 + \frac{(x-\mu)^2}{\nu\sigma^2}\right)^{-\frac{\nu+1}{2}}$

其中$\nu > 0$为自由度参数，$\Gamma$为伽马函数。

**性质**：

- 当$\nu \to \infty$时退化为高斯分布
- $\nu$较小时有重尾(Heavy Tails)，对异常值(Outliers)鲁棒
- 在计算机视觉中用于鲁棒估计

---

**伯努利分布与多项式分布**

伯努利分布： $p(x|p) = p^x(1-p)^{1-x}, \quad x \in {0, 1}$

多项式分布： $p(\mathbf{x}|\boldsymbol{p}) = \frac{n!}{x_1! x_2! \cdots x_k!} p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k}$

其中$\sum_i x_i = n$，$\sum_i p_i = 1$。

---

**冯·米塞斯分布(von Mises Distribution)**

用于建模方向数据(Directional Data)在$S^1$(单位圆)上的分布： $p(\theta|\mu, \kappa) = \frac{\exp(\kappa\cos(\theta-\mu))}{2\pi I_0(\kappa)}$

其中$\mu$为均值方向，$\kappa \geq 0$为浓度参数(Concentration Parameter)，$I_0$为修正贝塞尔函数。

**应用**：人体关键点、光流方向等循环数据的建模。

---

**1.3.3 参数估计**

**最大似然估计(Maximum Likelihood Estimation, MLE)**

给定数据$\mathcal{D} = {\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N}$，参数的MLE为： $\hat{\theta}_{\text{MLE}} = \arg\max_\theta p(\mathcal{D}|\theta) = \arg\max_\theta \prod_{i=1}^N p(\mathbf{x}_i|\theta)$

取对数以简化计算(因对数是单调递增函数)： $\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^N \log p(\mathbf{x}_i|\theta) = \arg\max_\theta \mathcal{L}(\theta)$

其中$\mathcal{L}(\theta) = \sum_{i=1}^N \log p(\mathbf{x}_i|\theta)$为对数似然(Log-Likelihood)。

**高斯MLE**

对于高斯数据$\mathbf{x}_i \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，MLE为： $\hat{\boldsymbol{\mu}} = \frac{1}{N}\sum_{i=1}^N \mathbf{x}_i$ $\hat{\boldsymbol{\Sigma}} = \frac{1}{N}\sum_{i=1}^N (\mathbf{x}_i - \hat{\boldsymbol{\mu}})(\mathbf{x}_i - \hat{\boldsymbol{\mu}})^T$

---

**最大后验估计(Maximum A Posteriori, MAP)**

结合先验信息的参数估计： $\hat{\theta}_{\text{MAP}} = \arg\max_\theta p(\theta|\mathcal{D}) = \arg\max_\theta \frac{p(\mathcal{D}|\theta)p(\theta)}{p(\mathcal{D})}$

由于$p(\mathcal{D})$与$\theta$无关，可简化为： $\hat{\theta}_{\text{MAP}} = \arg\max_\theta [p(\mathcal{D}|\theta)p(\theta)] = \arg\max_\theta [\mathcal{L}(\theta) + \log p(\theta)]$

**与正则化的联系**：

若取负对数似然与负对数先验： $\hat{\theta}_{\text{MAP}} = \arg\min_\theta [-\mathcal{L}(\theta) - \log p(\theta)]$

这等价于在最小二乘中加入正则项。例如，若$p(\theta) \propto \exp(-\lambda|\theta|_2^2)$，则： $\hat{\theta}_{\text{MAP}} = \arg\min_\theta [\text{数据项} + \lambda|\theta|_2^2]$

这正是L2正则化(岭回归)。

---

**期望最大算法(Expectation-Maximization, EM)**

用于含隐变量的参数估计。设观测数据为$\mathcal{D} = {\mathbf{x}_1, \ldots, \mathbf{x}_N}$，隐变量为$\mathbf{Z} = {\mathbf{z}_1, \ldots, \mathbf{z}_N}$。

完全数据的对数似然： $\mathcal{L}_{\text{complete}}(\theta) = \log p(\mathcal{D}, \mathbf{Z}|\theta)$

**E步(Expectation Step)**： $Q(\theta, \theta^{(t)}) = \mathbb{E}_{\mathbf{Z}|\mathcal{D}, \theta^{(t)}}[\mathcal{L}_{\text{complete}}(\theta)]$

计算关于隐变量后验的期望对数似然。

**M步(Maximization Step)**： $\theta^{(t+1)} = \arg\max_\theta Q(\theta, \theta^{(t)})$

**收敛性**：EM算法单调增加观测数据的似然，即$\mathcal{L}(\theta^{(t+1)}) \geq \mathcal{L}(\theta^{(t)})$。

**在CV中的应用**：高斯混合模型(GMM)、无监督聚类等。

---

**1.3.4 图模型(Graphical Models)**

**贝叶斯网络(Bayesian Networks)**

有向无环图(DAG)：顶点为随机变量，有向边表示条件依赖关系。

联合分布的因式分解： $p(\mathbf{x}_1, \ldots, \mathbf{x}_n) = \prod_{i=1}^n p(x_i|\mathbf{pa}_i)$

其中$\mathbf{pa}_i$为节点$i$的父节点集合。

**条件独立性**：若变量$X$和$Y$被变量集合$Z$d-分离(d-separated)，则$X \perp Y | Z$。

---

**马尔可夫随机场(Markov Random Field, MRF)**

无向图表示的概率模型。联合分布表示为： $p(\mathbf{x}) = \frac{1}{Z}\prod_{c \in \mathcal{C}} \phi_c(\mathbf{x}_c)$

其中$\mathcal{C}$为团(Cliques)的集合，$\phi_c$为非负势函数(Potential Function)，$Z$为配分函数(Partition Function)： $Z = \sum_{\mathbf{x}} \prod_{c \in \mathcal{C}} \phi_c(\mathbf{x}_c)$

通常取$\phi_c(\mathbf{x}_c) = \exp(-E(\mathbf{x}_c))$，则： $p(\mathbf{x}) = \frac{1}{Z}\exp\left(-\sum_{c \in \mathcal{C}} E(\mathbf{x}_c)\right)$

其中$E(\mathbf{x})$为能量函数(Energy Function)。

**Hammersley-Clifford定理**：MRF的Markov性质和Gibbs分布(上式)等价。

---

**条件随机场(Conditional Random Field, CRF)**

用于条件概率$p(\mathbf{y}|\mathbf{x})$的无向图模型： $p(\mathbf{y}|\mathbf{x}) = \frac{1}{Z(\mathbf{x})}\exp\left(\sum_{i,j} \theta_{ij}(x_i, y_i, y_j)\right)$

其中$Z(\mathbf{x})$为依赖于观测$\mathbf{x}$的配分函数： $Z(\mathbf{x}) = \sum_{\mathbf{y}} \exp\left(\sum_{i,j} \theta_{ij}(x_i, y_i, y_j)\right)$

**线性CRF(特例)**： $p(\mathbf{y}|\mathbf{x}) = \frac{1}{Z(\mathbf{x})}\exp\left(\sum_i \mathbf{w}_i^T \mathbf{f}_i(x_i, y_i) + \sum_{i,j} \mathbf{w}_{ij}^T \mathbf{f}_{ij}(y_i, y_j)\right)$

**应用**：序列标注(序列分割)、图像分割等。

---

**因子图与信念传播(Belief Propagation)**

因子图为二部图，分离变量节点与因子节点。

**求和-乘积算法(Sum-Product Algorithm)**：

对于树结构的图，可精确计算边际概率。对于一般图，循环信念传播(Loopy BP)是近似方法。

消息定义：

- 变量$x_i$到因子$f$的消息：$\mu_{x_i \to f}(x_i) \propto \prod_{f' \in \text{nb}(x_i) \setminus f} \mu_{f' \to x_i}(x_i)$
- 因子$f$到变量$x_i$的消息：$\mu_{f \to x_i}(x_i) \propto \sum_{\mathbf{x} \setminus x_i} f(\mathbf{x}) \prod_{x_j \in \text{nb}(f) \setminus x_i} \mu_{x_j \to f}(x_j)$

---

**1.3.5 变分推断(Variational Inference)**

对于难以处理的后验分布$p(\mathbf{z}|\mathbf{x})$，用易处理的分布$q(\mathbf{z})$近似。

**变分下界(Evidence Lower Bound, ELBO)**

观测数据的对数似然可分解为： $\log p(\mathbf{x}) = \text{ELBO}(q) + D_{KL}(q(\mathbf{z})|p(\mathbf{z}|\mathbf{x}))$

其中ELBO定义为： $\text{ELBO}(q) = \mathbb{E}_q[\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_q[\log q(\mathbf{z})]$

$= \mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z})] - D_{KL}(q(\mathbf{z})|p(\mathbf{z}))$

由于KL散度非负，ELBO为$\log p(\mathbf{x})$的下界。最大化ELBO等价于最小化KL散度。

**KL散度(Kullback-Leibler Divergence)**

$D_{KL}(q|p) = \mathbb{E}_q[\log q(\mathbf{z}) - \log p(\mathbf{z}|\mathbf{x})] = \int q(\mathbf{z})\log\frac{q(\mathbf{z})}{p(\mathbf{z}|\mathbf{x})}d\mathbf{z}$

性质：$D_{KL}(q|p) \geq 0$，等号成立当且仅当$q = p$。

---

**平均场近似(Mean Field Approximation)**

假设$q(\mathbf{z})$可分解： $q(\mathbf{z}) = \prod_{i=1}^m q_i(z_i)$

**坐标上升(Coordinate Ascent)**更新： $q_j^*(z_j) \propto \exp(\mathbb{E}_{q_{-j}}[\log p(\mathbf{x}, \mathbf{z})])$

其中$q_{-j} = \prod_{i \neq j} q_i(z_i)$。

---

#### 1.4 几何学基础(Geometric Foundations)

**1.4.1 欧氏几何(Euclidean Geometry)**

**向量与点**

在$\mathbb{R}^3$中，点$P$用坐标$\mathbf{p} = (x, y, z)^T$表示，向量$\mathbf{v}$为两点差$\mathbf{v} = Q - P$。

**内积与外积**

内积(点积)： $\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T\mathbf{v} = \sum_i u_i v_i = |\mathbf{u}||\mathbf{v}|\cos\theta$

其中$\theta$为两向量夹角。

外积(叉积)： $\mathbf{u} \times \mathbf{v} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \ u_1 & u_2 & u_3 \ v_1 & v_2 & v_3 \end{vmatrix}$

性质：垂直于$\mathbf{u}$和$\mathbf{v}$，大小为$|\mathbf{u}\times\mathbf{v}| = |\mathbf{u}||\mathbf{v}|\sin\theta$。

反对称矩阵形式： $[\mathbf{u}]_\times = \begin{pmatrix} 0 & -u_3 & u_2 \ u_3 & 0 & -u_1 \ -u_2 & u_1 & 0 \end{pmatrix}$

满足$[\mathbf{u}]_\times \mathbf{v} = \mathbf{u} \times \mathbf{v}$。

---

**坐标变换(Coordinate Transformation)**

刚体变换(旋转+平移)： $\mathbf{p}' = R\mathbf{p} + \mathbf{t}$

其中$R \in SO(3)$为旋转矩阵，$\mathbf{t} \in \mathbb{R}^3$为平移向量。

$SO(3)$的性质：

- $R^T R = I$（正交性）
- $\det(R) = 1$（保向）
- 元素个数：9，自由度：3(欧拉角或轴角)

---

**距离与角度**

欧氏距离： $d(\mathbf{p}, \mathbf{q}) = |\mathbf{p} - \mathbf{q}|_2 = \sqrt{\sum_i (p_i - q_i)^2}$

两向量夹角： $\cos\theta = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{|\mathbf{u}||\mathbf{v}|}$

---

**1.4.2 射影几何(Projective Geometry)**

**齐次坐标(Homogeneous Coordinates)**

点$P = (x, y)$在平面上的齐次坐标为$\tilde{\mathbf{p}} = (x, y, 1)^T \in \mathbb{P}^2$(射影平面)。

一般地，$\mathbf{p} = \lambda(x, y, 1)^T$对任意$\lambda \neq 0$表示同一点。

直线的齐次坐标：$\mathbf{l} = (a, b, c)^T$表示直线$ax + by + c = 0$。

点在直线上：$\mathbf{l}^T\mathbf{p} = 0$。

**对偶性**：点与直线的关系具有对偶性。

---

**射影变换(Projective Transformation)**

齐次坐标下的线性变换： $\tilde{\mathbf{p}}' = H\tilde{\mathbf{p}}$

其中$H \in GL(3)$(可逆$3 \times 3$矩阵)，$\lambda H$与$H$表示同一射影变换($\lambda \neq 0$)，故有8个自由度。

矩阵形式： $H = \begin{pmatrix} h_{11} & h_{12} & h_{13} \ h_{21} & h_{22} & h_{23} \ h_{31} & h_{32} & h_{33} \end{pmatrix}$

**无穷远线(Line at Infinity)**

齐次坐标中，$\mathbf{l}_\infty = (0, 0, 1)^T$表示无穷远线。射影变换可改变无穷远线。

**交比(Cross-Ratio)**

四点$P_1, P_2, P_3, P_4$在直线上，其交比定义为： $(P_1, P_2; P_3, P_4) = \frac{|P_1 P_3| / |P_2 P_3|}{|P_1 P_4| / |P_2 P_4|}$

交比在射影变换下不变。

---

**1.4.3 黎曼几何(Riemannian Geometry)**

**黎曼流形(Riemannian Manifold)**

光滑流形$M$配备黎曼度量$g$，即在每点$p \in M$的切空间$T_p M$上定义内积$g_p(\cdot, \cdot)$。

度量张量(Metric Tensor)坐标表示： $g = \sum_{i,j} g_{ij}dx^i \otimes dx^j$

弧长元： $ds = \sqrt{g_{ij}dx^i dx^j}$

---

**测地线(Geodesic)**

黎曼流形上连接两点的最短路径。参数化曲线$\gamma(t)$为测地线当且仅当： $\frac{D}{\text{d}t}\dot{\gamma} = 0$

其中$D/\text{d}t$为协变导数(Covariant Derivative)。

在局部坐标中： $\ddot{\gamma}^k + \Gamma^k_{ij}\dot{\gamma}^i\dot{\gamma}^j = 0$

其中$\Gamma^k_{ij}$为Christoffel符号。

**应用**：$SO(3)$(旋转群)上的测地距离、特征空间的流形学习。

---

**曲率(Curvature)**

黎曼曲率张量： $R^l_{ijk} = \frac{\partial \Gamma^l_{ik}}{\partial x^j} - \frac{\partial \Gamma^l_{ij}}{\partial x^k} + \Gamma^l_{mj}\Gamma^m_{ik} - \Gamma^l_{mk}\Gamma^m_{ij}$

Ricci曲率张量： $R_{ij} = R^k_{ikj}$

标量曲率： $R = g^{ij}R_{ij}$

**几何意义**：曲率描述流形偏离欧氏空间的程度。

---

**1.4.4 仿射几何(Affine Geometry)**

**仿射变换(Affine Transformation)**

形式： $\mathbf{x}' = A\mathbf{x} + \mathbf{b}$

其中$A \in GL(n)$，$\mathbf{b} \in \mathbb{R}^n$。自由度：$n^2 + n$。

**仿射不变量**：

- 平行线保持平行
- 平行线段的比例保持
- 体积比例(缩放$|\det(A)|$倍)

齐次坐标表示： $\begin{pmatrix} \mathbf{x}' \ 1 \end{pmatrix} = \begin{pmatrix} A & \mathbf{b} \ \mathbf{0}^T & 1 \end{pmatrix} \begin{pmatrix} \mathbf{x} \ 1 \end{pmatrix}$

---

**仿射相机模型**

对比完全射影相机$P = K[R|\mathbf{t}]$，仿射相机忽略深度变化的影响。3D点$(X, Y, Z)^T$到2D像素$(x, y)^T$的映射为： $\begin{pmatrix} x \ y \end{pmatrix} = M\begin{pmatrix} X \ Y \ Z \end{pmatrix} + \mathbf{b}$

其中$M \in \mathbb{R}^{2 \times 3}$，$\mathbf{b} \in \mathbb{R}^2$。

**应用**：弱透视投影、小角近似等简化模型。

---

本部分第一部分(基础理论)正式完成。内容涵盖：

- 线性代数完整理论(SVD、QR等)
- 优化理论(梯度法、牛顿法、KKT条件)
- 概率论与推断(MLE、MAP、EM、图模型、变分推断)
- 几何学(欧氏、射影、黎曼、仿射)
