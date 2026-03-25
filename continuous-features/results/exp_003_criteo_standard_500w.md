# exp_003：Criteo Standard，17种编码器，500万样本

## 实验信息

| 项目 | 内容 |
|------|------|
| 日期 | 2026-03-25 |
| 数据集 | Criteo Display Advertising（标准版） |
| 样本量 | 500万（train=400万，test=100万） |
| 连续特征数 | 13（I1-I13） |
| 类别特征数 | 26（C1-C26） |
| Backbone | DeepFM（FM + MLP(256,128,64)） |
| Epochs | 1 |
| Batch size | 4096 |
| 设备 | CPU |

## 实验结果

| 方法 | AUC | 参数量 | 训练时间 |
|------|-----|--------|---------|
| NoneEncoder（ablation） | 0.7579 | 4568.9K | 124s |
| ScalarEncoder（baseline） | 0.7893 | 4572.4K | 62s |
| BucketEncoder | 0.7900 | 4627.9K | 52s |
| AutoDisEncoder | 0.7901 | 4644.2K | 56s |
| NumericEmbedding | 0.7915 | 4640.8K | 57s |
| FTTransformer | 0.7890 | 4708.7K | 89s |
| PeriodicEncoder | 0.7893 | 4682.8K | 38s |
| FieldEmbedding | 0.7906 | 4625.9K | 35s |
| DLRMEncoder | 0.7911 | 4579.3K | 34s |
| MinMaxEncoder | 0.7912 | 4572.4K | 33s |
| StandardScalerEncoder | 0.7895 | 4572.4K | 35s |
| LogTransformEncoder | 0.7886 | 4572.4K | 32s |
| NumericEmbeddingDeep | 0.7917 | 4694.9K | 44s |
| **NumericEmbeddingSiLU** | **0.7918** | 4640.8K | 42s |
| NumericEmbeddingLN | 0.7872 | 4642.5K | 42s |
| NumericEmbeddingContextual | 0.7917 | 4641.0K | 42s |
| PLREncoder | 0.7909 | 4629.6K | 41s |

## 结论分析

### 排名
1. **NumericEmbeddingSiLU（0.7918）** — 最优，SiLU 激活函数效果更好
2. NumericEmbeddingDeep / NumericEmbeddingContextual（0.7917）— 并列第二
3. NumericEmbedding（0.7915）— 原始版本仍然很强
4. AutoDisEncoder（0.7901）/ BucketEncoder（0.7900）— 离散化方法次之
5. FTTransformer（0.7890）— 效果不如 Numeric 系列，且训练最慢（89s），**不推荐**

### 连续特征贡献
- NoneEncoder（无连续特征）vs NumericEmbeddingSiLU：**+33.9 千分点**
- 连续特征对 CTR 预估贡献极大（13个特征 > 8个特征场景）

### 性价比分析
- **最优效果**：NumericEmbeddingSiLU（0.7918，42s）
- **最佳性价比**：DLRMEncoder（0.7911，34s）/ MinMaxEncoder（0.7912，33s）
- **不推荐**：FTTransformer（效果差+速度慢）、NumericEmbeddingLN（AUC最低的Numeric变体）

### 与100万样本对比（exp_002）
| 方法 | 100万 AUC | 500万 AUC | 涨幅 |
|------|-----------|-----------|------|
| NoneEncoder | 0.7422 | 0.7579 | +15.7千分点 |
| ScalarEncoder | 0.7759 | 0.7893 | +13.4千分点 |
| NumericEmbedding | 0.7792 | 0.7915 | +12.3千分点 |
| FTTransformer | 0.7782 | 0.7890 | +10.8千分点 |
- 数据量从100万→500万，各方法普遍涨 10-16 千分点
- 方法间相对排名基本稳定，NumericEmbedding 系列持续领先

## 超参数
- embedding_dim: 16
- MLP: (256, 128, 64)
- lr: 1e-3
- cat_vocab_size: 10000
- seed: 42
