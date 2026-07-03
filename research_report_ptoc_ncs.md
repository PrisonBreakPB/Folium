# 网络控制下的预设时间最优控制问题：文献调研报告

> 调研时间：2026-06-27  
> 数据来源：arXiv (arxiv Python API)  
> 检索论文数：约 60 篇，精读约 15 篇

---

## 1. 概念背景

### 1.1 什么是预设时间控制 (Prescribed-Time Control, PTC)？

预设时间控制由 **Song Yongduan** 等人首创（Ye, Song & Lewis, 2022），其核心特征是：

> 系统状态在**用户预先指定的有限时间内**收敛到期望值，该收敛时间**与初始条件无关**，可任意设定。

**与相关概念的区别**：

| 概念 | 收敛时间 | 与初始值关系 | 时间可预先指定？ |
|------|---------|-------------|:---:|
| 渐近稳定 (Asymptotic) | t -> inf | 无关 | -- |
| 有限时间 (Finite-Time) | 有限 | 有关 | No |
| 固定时间 (Fixed-Time) | 有限+有界 | 无关(有上界) | No (只能调参数) |
| **预设时间 (Prescribed-Time)** | 有限 | 无关 | **Yes 直接设定** |
| 预定义时间 (Predefined-Time) | 有限+上界 | 无关 | 设定上界 |

> 注意：Prescribed-time（可精确设定）与 Predefined-time（设定上界保证在此之前收敛）有细微区别（Aldana-Lopez et al., 2023），但文献中常有混用。

---

## 2. 核心方法论

### 2.1 时域映射法 (Time-Scale Transformation)

**代表工作**：Shakouri & Assadian (2021, 2019)

核心思想：通过一个奇异映射函数 $\phi: [0, T) \to [0, \infty)$，将有限区间 $[0, T)$ 映射到无限区间 $[0, \infty)$，从而将预设时间控制问题转化为**渐近控制问题**。

- 优点：框架统一，可用于带约束的非线性系统
- 缺点：终端时刻增益趋于无穷（奇异性问题）

### 2.2 时变增益法 (Time-Varying Gains)

**代表工作**：Song et al.; Ye, Song & Lewis (2022)

核心思想：构造时变增益函数 $\rho(t) = \frac{T}{T - t}$ 型结构，在 $t \to T$ 时驱动增益趋于无穷，保证 $T$ 时刻收敛。

- 优点：设计直观，鲁棒性强（可抗未知上界扰动）
- 缺点：增益奇异（对量化噪声/测量噪声敏感）

### 2.3 有界增益的预定义时间控制

**代表工作**：Aldana-Lopez, Seeber, Haimovich & Gomez-Gutierrez (2023)

提出保持**增益一致有界**的预定义时间控制器设计方法，克服了奇异增益的工程缺陷。通过条件 Lyapunov 稳定性分析保证收敛时间上界。

---

## 3. 网络控制 + 预设时间：交叉研究现状

这是本调研的核心关注点。我将文献组织为以下四个子方向：

### 3.1 分布式预设时间优化

| 论文 | 年份 | 内容 |
|------|:---:|------|
| Zuo, Li & Zhu | 2024 | 网络化 Euler-Lagrange 系统的分布式预设时间凸优化 (DPTCO)，基于位置相关梯度 + 邻居信息交互 |
| Zuo, Zhu, Wang & Chen | 2024 | 线性异构多智能体系统的预设时间协同输出调节 |
| Zuo, Zhu & Wang | 2024 | 高阶非线性 MAS 的预设时间内分布式凸优化 |
| Gong, Li & Xu | 2024 | 带动态事件触发通信的预设时间收敛分布式多目标优化 |
| Zhang, Guo & Zhou | 2024 | 基于零梯度和滑模的预定义时间分布式优化 |

**关键发现**：这一方向文献最丰富。核心挑战是在**网络通信约束**（无向连通图、邻居信息交互）下，实现分布式优化目标函数的预设时间收敛。

### 3.2 事件触发 + 预设时间控制

| 论文 | 年份 | 核心贡献 |
|------|:---:|---------|
| Kashyap, Karnan & Jagtap | 2025 | 周期性事件触发自适应 barrier 控制，Euler-Lagrange 系统带状态/输入/时间约束 |
| Sunny & Warier | 2025 | 矩阵缩放多智能体网络的预设时间事件触发控制，状态依赖触发函数降低通信频率 |
| Gong, Li & Xu | 2024 | 动态事件触发通信的分布式多目标优化 |

**关键发现**：事件触发机制是网络控制下 PT 控制的核心方案——在减少通信次数的同时保证预设时间收敛。

### 3.3 博弈论与经济调度

| 论文 | 年份 | 内容 |
|------|:---:|------|
| Feng & Hu | 2020 | 非合作博弈中预设时间完全分布式 Nash 均衡寻求（固定/切换拓扑） |

### 3.4 多智能体一致性

| 论文 | 年份 | 内容 |
|------|:---:|------|
| Ye, Wen & Song | 2024 | 未知传感器灵敏度下的异构 MAS 预设时间领航-跟随一致性（分布式矩阵束方法） |
| Liu, Yang & Zhang | 2026 | 基于自适应预设时间 CBF 的多机器人网络时空重连 |

---

## 4. 关键空白："预设时间 + 最优控制"

### 4.1 已有工作

在 arxiv 上用 `"prescribed-time" AND "optimal control" AND "cost function"` 检索**几乎无结果**。

现有"PT + 最优"的工作实质上是：
- **分布式优化**在预设时间内求解（Zuo 等系列工作）
- 而非传统意义上的**最优控制**（最小化二次型指标、求解 HJB 方程、Pontryagin 极值原理）

### 4.2 核心空白

| 问题 | 现状 |
|------|------|
| PT 框架下的 **LQR 设计** | 未发现 |
| PT 框架下的 **模型预测控制 (MPC)** | 仅发现约束相关（非 PT-MPC） |
| 网络攻击下 (DoS/欺骗) 的 PT 控制 | 未发现 |
| 含通信时延/丢包的 PT 稳定性 | 未发现直接文献 |
| PT 控制的**最优性能指标** | 未发现 |
| PT 框架 + 强化学习 | 仅零星（H. Rahimi Nohooji 2025 用 RL 但非 PT+RL 结合） |

### 4.3 为什么是困难的问题？

1. **奇异性**：传统 PT 控制在终端时刻增益奇异，导致控制能量 $J = \int_0^T \|u(t)\|^2 dt$ 可能发散，无法定义标准二次型最优指标
2. **非线性**：PT 控制本质上是时变非线性控制，Bellman 最优性原理的时不变版本不直接适用
3. **网络不确定性**：时延、丢包、量化等网络非理想因素破坏了精确的时间映射

---

## 5. 代表性文献清单

### 综述

| # | 文献 |
|---|------|
| 1 | **Ye H., Song Y., Lewis F.L.** (2022). *Prescribed-Time Control and Its Latest Developments.* arXiv:2210.12712 |

### 核心方法论

| # | 文献 |
|---|------|
| 2 | **Aldana-Lopez R., Seeber R., Haimovich H., Gomez-Gutierrez D.** (2023). *Designing controllers with predefined convergence-time bound using bounded time-varying gains.* arXiv:2311.02473 |
| 3 | **Shakouri A., Assadian N.** (2021). *A Framework for Prescribed-Time Control Design via Time-Scale Transformation.* arXiv:2112.08496 |
| 4 | **Shakouri A., Assadian N.** (2019). *Prescribed-time Control for Perturbed Euler-Lagrange Systems with Obstacle Avoidance.* arXiv:1910.08529 |

### 网络控制 + 预设时间

| # | 文献 |
|---|------|
| 5 | **Zuo G., Li M., Zhu L.** (2024). *Small-Gain Theorem Based Distributed Prescribed-Time Convex Optimization For Networked Euler-Lagrange Systems.* arXiv:2407.19496 |
| 6 | **Zuo G., Zhu L., Wang Y., Chen Z.** (2024). *Prescribed-time Cooperative Output Regulation of Linear Heterogeneous Multi-agent Systems.* arXiv:2407.11408 |
| 7 | **Zuo G., Zhu L., Wang Y.** (2024). *Achieving distributed convex optimization within prescribed time for high-order nonlinear multiagent systems.* |
| 8 | **Kashyap C.S., Karnan A., Jagtap P., Keshavan J.** (2025). *Periodic Event-Triggered Prescribed Time Control of Euler-Lagrange Systems under State and Input Constraints.* arXiv:2510.02769 |
| 9 | **Sunny K.P., Warier R.R.** (2025). *Prescribed-Time Event-Triggered Control for Matrix-Scaled Networks.* arXiv:2509.07703 |
| 10 | **Gong T., Li Z., Xu Y.** (2024). *Prescribed-Time Convergent Distributed Multiobjective Optimization With Dynamic Event-Triggered Communication.* |
| 11 | **Feng Z., Hu G.** (2020). *Prescribed-Time Fully Distributed Nash Equilibrium Seeking in Noncooperative Games.* arXiv:2009.11649 |
| 12 | **Ye H., Wen C., Song Y.** (2024). *Distributed Matrix Pencil Formulations for Prescribed-Time Leader-Following Consensus of MASs with Unknown Sensor Sensitivity.* |
| 13 | **Zhang R., Guo G., Zhou Z.** (2024). *Balance of Communication and Convergence: Predefined-time Distributed Optimization Based on Zero-Gradient and Sliding Mode.* |

---

## 6. 未来研究建议

基于文献调研，**"网络控制下的预设时间最优控制"** 至少有以下几个有前景的方向：

1. **非奇异增益下的 PT 最优控制**
   - 基于 Aldana-Lopez (2023) 的 bounded gains 框架，设计使 LQR 型指标最小化的预设时间控制器
   - 挑战：时变 Riccati 方程的 PT 终端条件

2. **网络攻击弹性 PT 控制**
   - DoS 攻击导致通信中断时如何保持 PT 收敛？
   - 当前文献几乎空白

3. **PT 模型预测控制 (PT-MPC)**
   - 在每个滚动时域内嵌入 PT 终端约束
   - 结合事件触发通信

4. **通信高效的 PT 分布式优化**
   - 已有事件触发工作（Sunny 2025, Gong 2024），可进一步加入量化/带宽约束

5. **基于强化学习的 PT 最优控制**
   - 利用 RL 逼近时变 HJB 方程的解，实现在线 PT 最优策略

---

## 参考文献

*以上文献均来自 arXiv，通过 `arxiv` Python API 检索获得。报告生成于 2026-06-27。*
