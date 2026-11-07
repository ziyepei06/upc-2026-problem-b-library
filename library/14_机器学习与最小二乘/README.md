# 14 机器学习与最小二乘（15 篇）

这一格回答一个具体问题：**当最小二乘不好用的时候（观测只有两帧、噪声大、参数可辨识性差），机器学习能怎么帮？**

按用途分四组：

| 组 | 代表文献 | 能解决什么 |
|---|---|---|
| A. 用 ML 学一个「求解器」 | OptNet、可微凸优化层、深度展开、学习型优化器 | 把最小二乘/优化层嵌进网络，端到端训练；或用展开的迭代网络代替手工调参的迭代求解 |
| B. 用 ML 反演参数 | 摊销优化、模拟推断（SBI）、PINN | 训练一个网络直接把「观测 → 参数」映射出来（amortized），或把物理方程写进损失来反演参数 |
| C. 用 ML 增强状态估计 | 深度卡尔曼滤波、KalmanNet、贝叶斯 KalmanNet、自适应 KalmanNet | 动力学模型只有部分已知时，用网络学过程噪声/增益，同时保留协方差与不确定度 |
| D. 具体领域的 ML 替代模型 | 弹道阻力系数曲线预测、学习型预条件子 | 直接学 Cd(M) 这类经验曲线；或用 GNN 学线性求解器的预条件子，加速最小二乘迭代 |

## 文献清单

| 文件 | 出处 | 研究了什么 | 对本题（2026 B）的用途 |
|---|---|---|---|
| `2015-ICML-深度卡尔曼滤波.pdf` | Krishnan et al., *Deep Kalman Filters*, ICML 2015 / arXiv:1511.05121 | 用变分自编码器把卡尔曼滤波推广成可学习的时序生成模型，可做反事实推断（状态被改造后轨迹会怎样） | 说明「滤波器的模型可以学」；反事实推断正好对应「如果初速/旋转变了，落点如何变」 |
| `2016-NeurIPS-学习型优化器（学梯度下降）.pdf` | Andrychowicz et al., *Learning to learn by gradient descent by gradient descent*, NeurIPS 2016 / arXiv:1606.04474 | 把优化算法本身当成学习目标，用 LSTM 学出更新规则，在同类问题上超过手调算法 | 「用数据学一个针对本类问题的最优迭代器」——若两帧反演用迭代法求解，可借鉴其思路 |
| `2017-NeurIPS-OptNet可微优化层.pdf` | Amos & Kolter, *OptNet*, NeurIPS 2017 / arXiv:1703.00443 | 把二次规划作为网络的一层，用隐函数/双层级方法精确求导，实现端到端训练 | 「最小二乘作为一层」的规范做法：先定义物理约束下的 LS 层，再在其上训练参数 |
| `2019-JCP-物理信息神经网络PINN.pdf` | Raissi et al., *Physics Informed Deep Learning (Part I)*, J. Comput. Phys. 2019 / arXiv:1711.10561 | 把 PDE/ODE 残差写进损失，网络既拟合数据又满足物理方程；可用于「数据驱动求解」与「数据驱动发现参数」 | 可作为「物理 + 数据」混合反演的引用；也是评阅人熟悉的引用点 |
| `2019-NeurIPS-可微凸优化层.pdf` | Agrawal et al., *Differentiable Convex Optimization Layers*, NeurIPS 2019 / arXiv:1910.12430 | 对 DCP 凸规划整体求导（cvxpylayers），把任意凸优化问题当网络层 | 若把「带约束的最小二乘」写成凸规划，可用它拿梯度做端到端标定 |
| `2020-PNAS-模拟推断前沿SBI.pdf` | Cranmer, Brehmer & Louppe, *The frontier of simulation-based inference*, PNAS 2020 / arXiv:1911.01429 | 综述「正演模拟器 → 反演参数」的方法家族（ABC、神经密度估计、likelihood-free） | 本题正是 SBI 的教科书场景：正演模型清楚、观测极少、反演参数；用来论证「为什么用机器学习做反演」 |
| `2022-TCI-深度展开与生成先验.pdf` | Wei et al., *Deep Unfolding with Normalizing Flow Priors for Inverse Problems*, IEEE TCI 2022 / arXiv:2107.02848 | 把近端梯度迭代展开成网络，用生成模型（正则化流）学习先验，端到端训练 | 「展开迭代 + 学习先验」是稳定病态反演的主流做法；用于讨论两帧反演如何加先验 |
| `2021-Bernoulli-浅层网络最小二乘外层权重.pdf` | Braun, Kohler, Langer & Walk, *Convergence rates for shallow neural networks learned by gradient descent*, Bernoulli / arXiv:2107.09550 | 理论分析：隐藏层随机初始化后，用**线性最小二乘**求输出层权重的网络达到 1/√n 收敛率 | 论证「最小二乘 + 网络」的组合有统计保证；适合写在方法节的理论依据 |
| `2022-Amos-摊销优化教程.pdf` | Amos, *Tutorial on amortized optimization*, arXiv:2202.00665 | 系统讲「用学习预测优化问题的解」，比传统求解快几个数量级；涵盖目标、训练与误差分析 | 「两帧 → 初速/旋转/落点」完全可以做成摊销求解器；这份是方法论主线参考 |
| `2022-IEEE-KalmanNet学习型卡尔曼滤波.pdf` | Revach et al., *KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics*, IEEE TSP 2022 / arXiv:2107.10043 | 把卡尔曼滤波里依赖噪声统计与模型知识的**增益计算**换成一个紧凑 RNN：模型部分已知、噪声统计未知时优于 EKF/UKF/粒子滤波，同时保持低复杂度与可解释性 | 本题正处在「动力学已知、噪声统计未知」的夹缝里：这份是不必手算 Q/R 就能拿到增益与协方差的原始出处（另两篇 KalmanNet 是它的后续） |
| `2023-IEEE-贝叶斯KalmanNet.pdf` | Dahan et al., *Bayesian KalmanNet*, IEEE TSP 2023 / arXiv:2309.03058 | 给 DNN 增强的卡尔曼滤波补上误差协方差估计，使学习型滤波器也能输出不确定度 | 本题的题眼是不确定度：这份说明学习型滤波器如何同时给出协方差 |
| `2023-ICASSP-自适应KalmanNet.pdf` | Ni, Revach & Shlezinger, *Adaptive KalmanNet*, ICASSP 2023 / arXiv:2309.07016 | 部分已知的状态空间模型下，用少量数据快速适配变化的噪声/模型参数 | 若写「模型参数未知时在线自适应」，可引这一篇 |
| `2024-VPINN用最小二乘优化变分网络.pdf` | Uriarte et al., *Optimizing Variational PINNs Using Least Squares*, arXiv:2407.20417 | 用线性最小二乘解网络最后一层权重，与梯度下降混合，改善 PINN 收敛并大幅降计算量 | 「最小二乘不是被替代，而是嵌进 ML 训练流程」的典型例子，正好回应「ML 优化最小二乘」这一表述 |
| `2025-机器学习预测弹道阻力系数曲线.pdf` | Jokela, *Transfer Learning for Predictive Custom Drag Modeling*, TinyComputer 2025 | 用迁移学习 + MLP 从 1039 个雷达实测弹道数据预测整条 Cd(M) 曲线，误差 3.15%，并与 PINN/Transformer/Neural ODE 对比 | 本题需要 Cd(Re/v) 的经验取值：这是「ML 直接学空气动力系数曲线」的直接先例 |
| `2026-学习型预条件子（图神经网络）.pdf` | Zhang et al., *Graph Neural Multilevel Preconditioners for Iterative Solvers*, arXiv:2607.28456 | 用图神经网络学 AMG 风格的预条件子，加速稀疏线性系统迭代求解 | 若最小二乘规模变大，可用「学预条件子」加速；作为数值加速方向的引用 |

## 怎么在论文里用

1. 只提两帧、参数多于约束时，最小二乘**不是唯一解法**：可以（a）加物理先验做正则化、（b）用摊销网络直接预测、（c）用学习型滤波器融合。
2. 引用顺序建议：`2022-Amos-摊销优化教程`（方法框架）→ `2020-PNAS-模拟推断`（反演问题的定位）→ `2022-IEEE-KalmanNet`（模型已知、噪声未知时的学习型滤波）→ `2024-VPINN`（LS 与 ML 结合的实例）→ `2023-IEEE-贝叶斯KalmanNet`（不确定度）→ `2017-OptNet` / `2019-可微凸优化层`（若要写「把 LS 做成可训练层」）。
3. 若只在讨论节用两段：一段讲「学习型逆问题求解」，一段讲「不确定度怎么从 ML 里拿到」。

## 没下载到的

- Springer EPJ B 2025, *Machine learning in parameter estimation of nonlinear systems*（用 Huber 损失神经网络替代 NLS，Lorenz 系统误差 <1.2% 而 NLS 超过 12%）：全文页有 JavaScript 校验，抓不到 PDF；链接 `https://doi.org/10.1140/epjb/s10051-025-00904-7`。
- IEEE TSP 2022 原版的排版版（DOI 10.1109/TSP.2022.3158588）在订阅墙内；本地收的是同一篇的 arXiv 版 `2107.10043`，内容完整（15 页）。
- *LISTA: Learning Fast Approximations of Sparse Coding*（Gregor & LeCun, ICML 2010）：无 arXiv 版，未收录。
