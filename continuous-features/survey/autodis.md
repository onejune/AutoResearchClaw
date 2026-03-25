# AutoDis

## 基本信息
- **论文**：[AutoDis: Automatic Discretization for Embedding Numerical Features in CTR Prediction](https://arxiv.org/abs/2012.08986)
- **会议/年份**：KDD 2021
- **作者/机构**：Huifeng Guo, Bo Chen, Ruiming Tang 等 / 华为诺亚方舟实验室

## 核心思路
AutoDis 提出了一种端到端可学习的数值特征离散化方法。与手动分桶不同，它为每个数值特征维护 H 个"元 embedding"（meta-embeddings），通过一个小型 MLP 将标量值映射为 H 个权重，再用 softmax 软分配后对元 embedding 加权求和，得到最终 embedding。整个过程完全可微，无需预先定义分桶边界。

## 方法细节

**三个核心模块：**

1. **自动离散化（Automatic Discretization）**：为每个数值特征定义 H 个可学习的元 embedding $\{e_h\}_{h=1}^H$。

2. **软权重分配（Soft Assignment）**：用 MLP 将标量 $x$ 映射到 H 维 logits，再经 softmax 得到软权重：
   $$\alpha = \text{softmax}(\text{MLP}(x) / \tau)$$
   其中 $\tau$ 为温度参数，控制分配的"硬度"。

3. **聚合（Aggregation）**：最终 embedding 为加权求和：
   $$e = \sum_{h=1}^H \alpha_h \cdot e_h$$

**整体公式：**
$$\text{AutoDis}(x) = \text{softmax}(\text{MLP}(x)) \cdot E$$
其中 $E \in \mathbb{R}^{H \times d}$ 为元 embedding 矩阵。

## 报告效果
- **数据集**：Criteo、Avazu、Malware
- **Criteo AUC**：~0.8112（对比 baseline DNN 的 0.8098）
- 在多个 CTR 预估数据集上均优于手动分桶、归一化等传统方法

## 优点
- 端到端可学习，无需人工设计分桶边界
- 软分配保留了数值特征的连续性信息
- 温度参数可控制离散化程度
- 参数量适中，计算效率较高

## 缺点 / 局限性
- 每个特征独立维护 H 个元 embedding，特征数量多时参数量线性增长
- 温度参数 $\tau$ 需要调优
- MLP 网络结构设计（层数、宽度）对效果有影响
- 对于稀疏数值特征（大量缺失值）的处理方式未深入讨论
