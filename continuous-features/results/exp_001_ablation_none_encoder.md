# 实验 001：连续特征处理方法对比 + NoneEncoder Ablation

## 基本信息
- **日期**：2026-03-25
- **数据集**：Criteo Conversion Logs（KDD 2014）
- **样本量**：100 万行
- **Backbone**：DeepFM（FM + MLP(256,128,64)）
- **Embedding dim**：16
- **Epochs**：3（early stopping, patience=2）
- **优化器**：Adam, lr=1e-3
- **Batch size**：4096
- **连续特征**：8 个（I1-I8）
- **类别特征**：9 个（C1-C9）

---

## 方法介绍

### NoneEncoder（Ablation）
不使用任何连续特征，模型只依赖类别特征。作为消融实验基准，用于量化连续特征对整体效果的贡献。

### ScalarEncoder（Baseline）
最简单的处理方式：对连续特征做 log1p 变换 + StandardScaler 标准化后，直接将标量值输入模型，不做任何 embedding。DeepFM 的 MLP 部分负责学习非线性变换。

### BucketEncoder
传统离散化方法：对每个连续特征做**等频分桶**（默认 10 个桶），将连续值映射为桶索引，再做 embedding lookup。桶边界由训练集数据分布决定，是工业界最常用的方法。

### AutoDisEncoder
**论文**：[AutoDis: Automatic Discretization for Embedding Numerical Features](https://arxiv.org/abs/2012.08986)（KDD 2021，阿里巴巴）

为每个连续特征预设 H 个"元 embedding"（meta-embeddings），用一个小网络将连续值映射为 H 个桶的**软权重**（softmax），最终 embedding = 软权重加权求和元 embedding。整个过程端到端可学习，避免了手动分桶的硬边界问题。

### NumericEmbeddingEncoder
为每个连续特征独立设计一个小 MLP（Linear → ReLU → Linear），直接将标量值映射为 embedding 向量。每个特征有自己专属的参数，表达能力强，实现简单，是 FIVES 等工作的简化版。

### FTTransformerEncoder
**论文**：[Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)（NeurIPS 2021，Yandex）

将每个连续特征视为一个 token：`token_i = x_i * W_i + b_i`，所有 token 拼接后输入 Transformer（2层，4头注意力）。Transformer 的自注意力机制能捕捉特征间的高阶交互，是表格数据上效果最强的方法之一，但计算开销大。

### PeriodicEncoder
**论文**：[On Embeddings for Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556)（NeurIPS 2022）

用可学习的 sin/cos 周期函数将标量映射到高维空间：
`x → [sin(w₁x+b₁), cos(w₁x+b₁), ..., sin(wₖx+bₖ), cos(wₖx+bₖ)]`
权重 w、偏置 b 均可学习。周期函数天然适合捕捉数值特征的周期性规律（如时间、价格区间等）。

---

## 实验结果

| 方法 | AUC | 参数量 | 训练时间 | 备注 |
|------|-----|--------|----------|------|
| NoneEncoder（ablation） | 0.8516 | 1609.2 K | 14.9s | 纯类别特征，无连续特征 |
| ScalarEncoder（baseline） | 0.8550 | 1611.4 K | 16.2s | 连续值直接输入 |
| BucketEncoder | 0.8567 | 1645.6 K | 19.1s | 等频分桶 + embedding |
| AutoDisEncoder | 0.8562 | 1655.6 K | 20.0s | 软分桶，端到端学习 |
| **NumericEmbedding** | **0.8572** | 1653.5 K | 18.8s | 每特征独立 MLP embedding |
| FTTransformer | 0.8563 | 1705.0 K | 37.1s | Transformer token 化 |
| PeriodicEncoder | 0.8554 | 1679.4 K | 17.9s | sin/cos 周期映射 |

---

## 结论与分析

1. **连续特征有效**：NoneEncoder（0.8516）vs ScalarEncoder（0.8550），连续特征带来 **+3.4 个千分点**提升
2. **各方法差距小**：所有编码器 AUC 在 0.8550~0.8572 之间，差距仅 2.2 个千分点（连续特征只有 8 个，信息量有限）
3. **NumericEmbedding 最优**：AUC 0.8572，参数量适中，速度快，性价比最好
4. **FTTransformer 不划算**：训练时间是其他方法 2 倍（37s vs 15-20s），AUC 无对应提升
5. **AutoDis 未显著优于 Scalar**：差距仅 1.2 个千分点，可能需要更大数据量才能体现优势

## 待验证
- [ ] 全量数据（1589 万行）下各方法差距是否扩大
- [ ] AutoDis 在更多连续特征（标准 Criteo 13 个）上的表现
- [ ] 与 Ali-CCP 数据集上的结果对比
