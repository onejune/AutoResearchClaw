# exp005: Contrastive ID Learning - 对比学习增强 ID 表示

## 对应 Paper

**Contrastive Learning for Sequential Recommendation / ID Representation**

- **会议**: SIGIR 2023 / WWW 2023 (相关方向)
- **代表工作**:
  - CL4SRec: Contrastive Learning for Sequential Recommendation (WWW 2021)
  - S^3-Rec: Self-Supervised Learning for Sequential Recommendation (CIKM 2020)
  - "Contrastive ID Learning for Large-Scale Recommendation" (ArXiv 2023)

## 核心思想

### 问题动机

传统 ID embedding 的问题：
- **监督信号稀疏**：只有点击/转化提供梯度
- **尾部 ID 训练不足**：样本少，embedding 学不好
- **语义相似性未利用**：相似 ID（如同类商品）的 embedding 应该接近

### 方法概述

用**对比学习**增强 ID 表示：

```
正样本对 (Similar IDs):
  - 同一 campaign 的不同 adid
  - 同一 advertiser 的不同 campaign
  - 被同一用户点击的不同物品
  
负样本对 (Random IDs):
  - 随机采样的不同 ID

对比损失 (InfoNCE):
  L_cl = -log[ exp(sim(pos_pair)/τ) / Σ exp(sim(all_pairs)/τ) ]
```

### 关键技术

#### 1. 正样本构造策略

| 策略 | 说明 | 示例 |
|------|------|------|
| **层次化正样本** | 同一粗粒度下的细粒度 ID | 同 campaign 的 adid |
| **行为正样本** | 被同一用户交互的 ID | 同一用户点击的 adid |
| **增强正样本** | Dropout / Mask 后的同一 ID | emb1 = Dropout(emb), emb2 = Dropout(emb) |
| **时间正样本** | 相邻时间窗口的 ID | 同一小时的热门 adid |

#### 2. 对比损失实现

```python
def contrastive_loss(anchor_emb, positive_emb, negative_embs, tau=0.5):
    # 余弦相似度
    pos_sim = cosine_similarity(anchor_emb, positive_emb) / tau
    
    neg_sims = [cosine_similarity(anchor_emb, neg) / tau 
                for neg in negative_embs]
    
    # InfoNCE loss
    log_prob = pos_sim - torch.logsumexp(torch.stack([pos_sim] + neg_sims))
    return -log_prob.mean()
```

#### 3. 联合训练

```python
# 主任务损失（CTR/CVR 预测）
L_main = BCEWithLogitsLoss(preds, labels)

# 对比损失
L_cl = contrastive_loss(anchor_emb, pos_emb, neg_embs)

# 总损失
L_total = L_main + λ * L_cl
```

### 本项目实现

简化版对比学习，使用**层次化正样本**：

```python
# 正样本：同一 business_type 内的 demand_pkgname
for bt in batch['business_type'].unique():
    ids_in_bt = batch[batch['business_type'] == bt]['demand_pkgname']
    if len(ids_in_bt) > 1:
        anchor = ids_in_bt[0]
        positive = ids_in_bt[1:]
        # 计算对比损失
```

## 预期收益

- **尾部 ID AUC 提升**：+1~3 千分点
- **表示质量提升**：相似 ID 的 embedding 更接近
- **鲁棒性增强**：对噪声和稀疏性更不敏感

## 实验设计

### Baseline
- 普通 WideDeep，只用交叉熵损失

### Contrastive
- WideDeep + 对比损失
- λ 消融：{0.01, 0.1, 0.5, 1.0}

### 评估重点
- 整体 AUC / PCOC
- 尾部 ID 分组效果（按频次分位）
- Embedding 可视化（t-SNE）

## 参考实现

- 本项目实现：`src/methods/contrastive_loss.py`（待创建）
