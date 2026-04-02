# exp003: Hierarchical Embeddings - 多粒度 ID 特征融合

## 对应 Paper

**Learning Hierarchical Embeddings for Large-Scale Recommender Systems**

- **会议**: CIKM 2021
- **作者**: (待补充完整引用)
- **类似工作**: 
  - "Hierarchical Representations for Efficient Large-Scale Recommendation" (WWW 2020)
  - "Multi-Granularity Embedding for Recommender Systems" (ArXiv 2022)

## 核心思想

### 问题动机

ID 类特征（如广告 ID、物品 ID）存在**多粒度信息**：

| 粒度 | 示例 | 特点 |
|------|------|------|
| **细粒度** | adid=12345 | 信息丰富，但稀疏，易过拟合 |
| **中粒度** | campaignid=789 | 平衡 |
| **粗粒度** | business_type=电商 | 密集，泛化强，但信息少 |

传统方法只用单一粒度（通常是细粒度），导致：
- 尾部 ID 训练不充分
- 新 ID 冷启动问题严重

### 方法概述

同时学习多粒度 embedding，用**门控网络动态融合**：

```python
# 细粒度 embedding（如 adid）
fine_emb = Embedding(adid)       # [batch, dim]

# 粗粒度 embedding（如 category）
coarse_emb = Embedding(category) # [batch, dim]

# 门控网络：决定融合比例
gate = sigmoid(MLP([fine_emb, coarse_emb]))  # [batch, 1]

# 动态融合
final_emb = gate * fine_emb + (1 - gate) * coarse_emb
```

### 关键技术

#### 1. 层次化特征对构造

手动或自动定义细粒度→粗粒度映射：

| 细粒度特征 | 粗粒度特征 | 说明 |
|------------|------------|------|
| demand_pkgname | business_type | 包名 → 业务类型 |
| offerid | adx | 广告位 → 广告平台 |
| adid | campaignid | 广告 ID → 计划 ID |

#### 2. 门控机制

门控网络学习每个样本的**最优融合权重**：

- **高频 ID**：gate → 1（偏向细粒度，保留细节）
- **低频 ID**：gate → 0（偏向粗粒度，增强泛化）
- **新 ID**：gate → 0（用粗粒度信息冷启动）

#### 3. 端到端训练

整个模型（包括门控网络）联合优化，无需预训练。

### 广告 ID 层级结构（IVR 数据集）

由细到粗：
```
campaignid → campaignsetid → offerid → demand_pkgname → business_type
   最细粒度                                              最粗粒度
```

本项目使用的层次化特征对：
- `campaignid` → `campaignsetid` (广告计划 → 广告组)
- `offerid` → `demand_pkgname` (offer → 包名)
- `demand_pkgname` → `business_type` (包名 → 业务类型)

### 本项目实现

```python
class HierarchicalEmbedding(nn.Module):
    def __init__(self, fine_vocab, coarse_vocab, dim=64):
        self.fine_embedding = Embedding(fine_vocab, dim)
        self.coarse_embedding = Embedding(coarse_vocab, dim)
        self.gate_network = Sequential(
            Linear(dim*2, 32), ReLU(),
            Linear(32, 1), Sigmoid()
        )
    
    def forward(self, fine_ids, coarse_ids):
        fine_emb = self.fine_embedding(fine_ids)
        coarse_emb = self.coarse_embedding(coarse_ids)
        gate = self.gate_network(concat(fine_emb, coarse_emb))
        return gate * fine_emb + (1 - gate) * coarse_emb
```

## 预期收益

- **整体 AUC**：持平或微升（+0~1 千分点）
- **尾部 BT AUC**：显著提升（+2~5 千分点，如 bt_16）
- **冷启动 ID**：泛化能力增强
- **可解释性**：门控权重反映 ID 置信度

## 实验设计

### Baseline
- 普通 WideDeep，只用细粒度 ID

### Hierarchical
- 细粒度 + 粗粒度，门控融合

### 评估重点
- 整体 AUC / PCOC
- **分 BT 结果**：重点关注样本少的 BT（如 bk/shein）
- **门控统计**：各特征的平均 gate 值（是否高频→细粒度）

## 参考实现

- 本项目实现：`src/models/hierarchical.py`
- 实验脚本：`scripts/run_exp003_hierarchical.py`
