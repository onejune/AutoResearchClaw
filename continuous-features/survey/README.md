# 连续特征处理方法综述

本目录收录了连续特征（数值特征）处理方法的文献综述，聚焦于 CTR 预估和表格数据深度学习场景。

## 方法总览

| 方法 | 会议/年份 | 核心思路 | 适用场景 |
|------|-----------|----------|----------|
| [AutoDis](autodis.md) | KDD 2021 | 可学习软分桶 + 元 embedding 加权 | CTR 预估 |
| [FIVES](fives.md) | KDD 2022 | NAS 搜索特征交互 + 分段线性 embedding | 大规模表格数据 |
| [FT-Transformer](ft_transformer.md) | NeurIPS 2021 | 特征 token 化 + Transformer 全局注意力 | 通用表格数据 |
| [TabNet](tabnet.md) | AAAI 2021 | 稀疏注意力逐步特征选择 | 可解释性场景 |
| [NODE](node.md) | ICLR 2020 | 可微分软决策树集成 | 数值特征为主 |
| [Periodic Activations](periodic_activations.md) | NeurIPS 2021/2022 | sin/cos 周期函数映射高维空间 | 通用数值特征 |
| [Numeric Embedding](numeric_embedding.md) | NeurIPS 2022 | MLP/PLE 将标量映射为 embedding | 通用数值特征 |

## 方法分类

### 按处理方式分类

**离散化类（Discretization-based）：**
- AutoDis：软分桶，可学习边界
- FIVES：分段线性编码
- BucketEncoder（本实验 baseline）：等频硬分桶

**Embedding 映射类（Embedding-based）：**
- Numeric Embedding：MLP 映射
- Periodic Activations：周期函数映射

**注意力/交互类（Attention-based）：**
- FT-Transformer：Transformer 全局注意力
- TabNet：稀疏注意力特征选择

**树结构类（Tree-based）：**
- NODE：可微分软决策树

### 按计算复杂度分类

| 方法 | 参数量 | 计算复杂度 | 训练速度 |
|------|--------|-----------|----------|
| ScalarEncoder | 极少 | O(n) | 极快 |
| BucketEncoder | 少 | O(n) | 快 |
| Periodic | 少 | O(nK) | 快 |
| NumericEmbedding | 中 | O(n·h·d) | 中 |
| AutoDis | 中 | O(n·H·d) | 中 |
| FT-Transformer | 多 | O(n²·d) | 慢 |

## 关键结论

1. **数值 embedding 普遍优于直接使用标量**：在大多数数据集上，将数值特征映射为 embedding 向量能显著提升模型效果。

2. **软分桶优于硬分桶**：AutoDis 等可学习的软分桶方法通常优于等频/等宽硬分桶，因为可以端到端优化。

3. **Transformer 效果好但计算贵**：FT-Transformer 在表格数据上效果优异，但计算复杂度高，在特征数多的工业场景需要权衡。

4. **周期编码参数效率高**：Periodic Encoding 以极少的参数（2K per feature）实现较好的效果，适合参数预算有限的场景。

5. **CTR 场景的特殊性**：CTR 预估中类别特征占主导，连续特征处理的提升空间相对有限（通常 AUC 提升 0.001-0.005）。

## 参考文献

1. Guo et al. (2021). AutoDis: Automatic Discretization for Embedding Numerical Features in CTR Prediction. KDD 2021.
2. Gu et al. (2022). FIVES: Feature Interaction Via Edge Search for Large-Scale Tabular Data. KDD 2022.
3. Gorishniy et al. (2021). Revisiting Deep Learning Models for Tabular Data. NeurIPS 2021.
4. Arık & Pfister (2021). TabNet: Attentive Interpretable Tabular Learning. AAAI 2021.
5. Popov et al. (2020). Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data. ICLR 2020.
6. Sitzmann et al. (2020). Implicit Neural Representations with Periodic Activation Functions. NeurIPS 2020.
7. Gorishniy et al. (2022). On the Embeddings of Numerical Features in Tabular Deep Learning. NeurIPS 2022.
