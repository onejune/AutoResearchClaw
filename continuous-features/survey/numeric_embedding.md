# Numeric Embedding（数值特征 Embedding）

## 基本信息
- **论文**：多篇相关工作，代表性综述见 [On Embeddings for Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556)
- **会议/年份**：NeurIPS 2022（综述论文）
- **作者/机构**：Yury Gorishniy, Ivan Rubachev, Artem Babenko / Yandex Research

## 核心思路
数值 Embedding 是一类将标量数值特征映射为高维向量的通用方法，核心思想是：与其直接使用原始标量值，不如通过可学习的映射函数将每个数值特征转化为 embedding 向量，从而让模型更容易学习特征的非线性变换。常见实现包括：MLP 映射、分段线性编码（PLE）、周期函数编码等。

## 方法细节

**方法一：MLP Embedding（本实验实现）**
每个特征独立一个小 MLP：
$$e_j = \text{MLP}_j(x_j) = W_2 \cdot \text{ReLU}(W_1 x_j + b_1) + b_2$$
其中 $W_1 \in \mathbb{R}^{h \times 1}$，$W_2 \in \mathbb{R}^{d \times h}$。

**方法二：分段线性编码（Piecewise Linear Encoding, PLE）**
将值域分为 T 段，每段用线性函数编码：
$$\phi_t(x) = \text{clip}\left(\frac{x - b_{t-1}}{b_t - b_{t-1}}, 0, 1\right)$$
最终 embedding：$e = \sum_t \phi_t(x) \cdot \mathbf{w}_t$

**方法三：周期函数编码（Periodic Embedding）**
$$e_j = [\sin(w_1 x_j + b_1), \cos(w_1 x_j + b_1), \ldots, \sin(w_K x_j + b_K), \cos(w_K x_j + b_K)]$$

**综述论文的关键发现：**
- 数值 embedding 在多数表格数据集上显著优于直接使用标量值
- PLE + MLP 组合效果最好
- 周期函数编码在某些数据集上表现优异

## 报告效果
- **数据集**：11 个公开表格数据集（California Housing、Adult、Higgs 等）
- PLE embedding 在 Higgs AUC：0.8932（优于无 embedding 的 0.8912）
- 在多数数据集上，数值 embedding 比直接使用标量提升 0.1-0.5% AUC

## 优点
- 实现简单，易于集成到任何深度学习框架
- 参数量适中，计算开销小
- 可以与任何骨干网络（DeepFM、Transformer 等）结合
- MLP 版本完全端到端可学习

## 缺点 / 局限性
- 每个特征独立 MLP，特征数多时参数量线性增长
- 需要调整 hidden_dim 和 embedding_dim
- 对于极稀疏的数值特征（大量缺失）效果可能下降
- 相比 AutoDis 等专门方法，缺乏对数值有序性的显式建模
