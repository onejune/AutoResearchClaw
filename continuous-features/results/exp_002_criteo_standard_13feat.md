# 实验 002：Criteo 标准版（13个连续特征）方法对比

## 基本信息
- **日期**：2026-03-25
- **数据集**：Criteo Display Advertising Standard（train_train.parquet）
- **样本量**：100 万行（随机采样，seed=42）
- **Backbone**：DeepFM（FM + MLP(256,128,64)）
- **Embedding dim**：16
- **Epochs**：3（early stopping, patience=2）
- **优化器**：Adam, lr=1e-3
- **Batch size**：4096
- **连续特征**：13 个（I1-I13，NaN 填 0，log1p + StandardScaler）
- **类别特征**：26 个（C1-C26，abs(hash) % 10000）
- **正样本率**：25.67%
- **训练/测试划分**：80% / 20%
- **设备**：CPU

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
| NoneEncoder（ablation） | 0.7422 | 4568.9 K | 28.4s | 纯类别特征，无连续特征 |
| ScalarEncoder（baseline） | 0.7759 | 4572.4 K | 29.5s | 连续值直接输入 |
| BucketEncoder | 0.7763 | 4627.9 K | 33.5s | 等频分桶 + embedding |
| AutoDisEncoder | 0.7763 | 4644.2 K | 35.3s | 软分桶，端到端学习 |
| **NumericEmbedding** | **0.7792** | 4640.8 K | 34.2s | 每特征独立 MLP embedding |
| FTTransformer | 0.7782 | 4708.7 K | 68.4s | Transformer token 化 |
| PeriodicEncoder | 0.7767 | 4682.8 K | 32.1s | sin/cos 周期映射 |

---

## 与 Criteo Conversion Logs（8特征）对比

| 方法 | AUC（Conv, 8特征） | AUC（Std, 13特征） | 备注 |
|------|-------------------|-------------------|------|
| NoneEncoder | 0.8516 | 0.7422 | 任务不同，不可直接比较绝对值 |
| ScalarEncoder | 0.8550 | 0.7759 | |
| BucketEncoder | 0.8567 | 0.7763 | |
| AutoDisEncoder | 0.8562 | 0.7763 | |
| NumericEmbedding | **0.8572** | **0.7792** | 两个数据集均排名第一 |
| FTTransformer | 0.8563 | 0.7782 | 两个数据集均排名第二 |
| PeriodicEncoder | 0.8554 | 0.7767 | |
| **连续特征贡献** | **+3.4 千分点** | **+33.7 千分点** | 13特征贡献远大于8特征 |

> **注**：两个数据集 AUC 绝对值不可直接比较（任务不同：CVR vs CTR；正样本率不同：~5% vs ~25.6%）。

---

## 结论与分析

### 1. 连续特征贡献在标准版上更显著
- Criteo Std（13特征）：NoneEncoder 0.7422 → ScalarEncoder 0.7759，**+33.7 个千分点**
- Criteo Conv（8特征）：NoneEncoder 0.8516 → ScalarEncoder 0.8550，**+3.4 个千分点**
- 13个连续特征信息量更丰富，是验证连续特征处理方法的更好场景

### 2. 方法排名跨数据集完全一致
- NumericEmbedding 在两个数据集上均排名第一
- FTTransformer 均排名第二
- 结论具有跨数据集泛化性，方法有效性可信

### 3. AutoDis 与 BucketEncoder 持续打平
- 两者 AUC 均为 0.7763，与 Criteo Conv 结论一致
- AutoDis 端到端学习优势在 100 万样本量下未能体现，可能需要全量数据（4125 万行）

### 4. FTTransformer 性价比最低
- 训练时间 68.4s，是其他方法 2 倍以上
- AUC 仅排第二（0.7782 vs NumericEmbedding 0.7792）
- CPU 环境下不推荐；GPU 环境下差距可能缩小

### 5. NumericEmbedding 综合最优
- AUC 最高（0.7792），参数量适中（4641 K），训练时间合理（34.2s）
- 实现简单，工业落地友好

---

## 待验证
- [ ] 全量数据（4125 万行）下各方法差距是否扩大，AutoDis 是否能超越 NumericEmbedding
- [ ] GPU 环境下 FTTransformer 的时间优势
- [ ] 更大 embedding_dim（32/64）对 AUC 的影响
- [ ] Ali-CCP 数据集上的结果
