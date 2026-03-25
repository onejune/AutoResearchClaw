# TabNet

## 基本信息
- **论文**：[TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
- **会议/年份**：AAAI 2021
- **作者/机构**：Sercan Ö. Arık, Tomas Pfister / Google Cloud AI

## 核心思路
TabNet 是一种专为表格数据设计的深度学习架构，核心创新是使用**稀疏注意力机制**在每个决策步骤中动态选择最相关的特征子集。这种逐步特征选择的方式类似于决策树的分裂过程，但完全可微分，同时提供了模型可解释性（哪些特征在哪个步骤被使用）。

## 方法细节

**整体架构（Sequential Multi-Step Decision）：**

TabNet 由 N 个决策步骤（decision steps）组成，每步包含：

1. **特征选择（Feature Selection）**：
   $$\mathbf{M}[i] = \text{sparsemax}(\mathbf{h}[i-1] \cdot \mathbf{W}_{\text{att}})$$
   其中 sparsemax 产生稀疏权重，$\mathbf{M}[i]$ 为特征掩码。

2. **特征处理（Feature Processing）**：
   $$\mathbf{h}[i] = f(\mathbf{M}[i] \odot \mathbf{x})$$
   使用 Batch Normalization + GLU（Gated Linear Unit）。

3. **累积输出**：
   $$\text{output} = \sum_{i=1}^{N} \text{ReLU}(\mathbf{h}[i] \cdot \mathbf{W}_i)$$

**稀疏正则化：**
$$L_{\text{sparse}} = \sum_{i=1}^{N} \sum_j \frac{-\mathbf{M}_{ij} \log(\mathbf{M}_{ij} + \epsilon)}{N \cdot B}$$

**数值特征处理：**
TabNet 直接使用原始数值特征（经 BN 归一化），不做额外 embedding，依赖注意力机制选择相关特征。

## 报告效果
- **数据集**：Forest Cover Type、KDD Census-Income、Poker Hand、Sarcos、Rossmann Store Sales
- Forest Cover Type 准确率：96.99%（优于 XGBoost 的 96.18%）
- 在多个数据集上优于 DNN、XGBoost 等基线
- 提供特征重要性可解释性

## 优点
- 可解释性强：可以可视化每个决策步骤使用的特征
- 稀疏特征选择减少不相关特征的干扰
- 支持无监督预训练（TabNet Encoder-Decoder）
- 参数效率较高（稀疏注意力）

## 缺点 / 局限性
- 训练不稳定，对学习率和 BN 动量较敏感
- 超参数较多（步骤数 N、稀疏系数等）
- 在 CTR 预估任务上效果不如专门设计的 FM 类模型
- 不擅长处理高基数类别特征（需要 embedding 预处理）
- 计算效率相对较低（多步骤串行处理）
