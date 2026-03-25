# FIVES

## 基本信息
- **论文**：[FIVES: Feature Interaction Via Edge Search for Large-Scale Tabular Data](https://arxiv.org/abs/2207.01943)
- **会议/年份**：KDD 2022
- **作者/机构**：Yulong Gu, Zhuoye Ding, Shuaiqiang Wang 等 / 京东

## 核心思路
FIVES 将特征交互建模为图上的边搜索问题，通过神经架构搜索（NAS）自动发现最优的特征交互组合。对于数值特征，FIVES 提出了一种基于分段线性函数的嵌入方法，将连续值映射到分段线性空间后再做 embedding，兼顾了数值特征的有序性和非线性表达能力。

## 方法细节

**数值特征处理（Piecewise Linear Embedding）：**

1. 将数值特征的值域划分为 K 个区间 $[b_{k-1}, b_k)$
2. 对于落在第 $k$ 个区间的值 $x$，计算分段线性编码：
   $$\phi_k(x) = \frac{x - b_{k-1}}{b_k - b_{k-1}}$$
3. 将分段编码与可学习 embedding 结合：
   $$e(x) = \sum_k \phi_k(x) \cdot W_k$$

**特征交互搜索：**
- 将特征交互建模为有向图，节点为特征，边为交互操作
- 通过可微 NAS 搜索最优交互边集合
- 候选操作包括：element-wise product、inner product、concat + MLP 等

## 报告效果
- **数据集**：Criteo、Avazu、KDD2012
- **Criteo AUC**：~0.8128（显著优于 DNN、DeepFM 等基线）
- 在工业级大规模数据集上验证了有效性

## 优点
- 自动搜索特征交互，减少人工特征工程
- 分段线性 embedding 保留数值特征的局部线性结构
- NAS 框架灵活，可扩展到不同交互操作
- 在大规模工业数据集上有实际验证

## 缺点 / 局限性
- NAS 搜索阶段计算成本较高
- 分段边界仍需预先定义（类似手动分桶）
- 模型复杂度较高，部署成本大
- 搜索空间设计对最终效果影响较大
