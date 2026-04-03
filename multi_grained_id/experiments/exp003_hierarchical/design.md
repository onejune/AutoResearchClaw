# exp003: Hierarchical Embeddings (CIKM 2021)

**日期**: 2026-04-01  
**状态**: 🔄 进行中

## 研究问题

在广告推荐场景中，ID 类特征（如 adid、demand_pkgname）通常具有：
- **高频头部**：少数 ID 占据大部分样本
- **长尾稀疏**：大量 ID 只有几个样本
- **动态更新**：新 ID 不断出现，旧 ID 失效

传统单一粒度 embedding 的问题：
- 细粒度（如 adid）：信息丰富但稀疏，易过拟合
- 粗粒度（如 category）：密集但信息少，欠拟合

**核心问题**：如何自动选择合适的粒度？

## 方法

### Hierarchical Embeddings (CIKM 2021)

同时学习细粒度和粗粒度 embedding，用门控网络动态融合：

```python
fine_emb = Embedding(adid)       # 细粒度
coarse_emb = Embedding(category) # 粗粒度

gate = sigmoid(MLP([fine_emb, coarse_emb]))
final_emb = gate * fine_emb + (1 - gate) * coarse_emb
```

**优势**：
- 端到端学习，无需人工规则
- 稀疏 ID 自动偏向粗粒度（泛化强）
- 密集 ID 自动偏向细粒度（信息丰富）

## 实验设计

### 层次化特征对

| 细粒度特征 | 粗粒度特征 | 说明 |
|------------|------------|------|
| demand_pkgname | business_type | 包名 → 业务类型 |
| offerid | adx | 广告位 → 广告交换平台 |

### 对比方案

| 模型 | 说明 |
|------|------|
| Baseline | 普通 WideDeep，只用细粒度 |
| Hierarchical | 层次化 Embedding，动态融合 |

### 评估指标

- 整体：AUC, PCOC, LogLoss
- 分 BT：重点关注 bt_16（样本少、分布漂移大）
- 门控分析：各特征的门控权重分布

## 假设

- **H1**: Hierarchical 在整体 AUC 上与 Baseline 持平或略优
- **H2**: Hierarchical 在 bt_16 上显著提升（粗粒度帮助泛化）
- **H3**: 门控权重与特征频次相关（低频→粗粒度，高频→细粒度）

## 预期结果

- bt_16 AUC 提升 1-2 千分点
- 整体 AUC 持平或微升
- 门控权重呈现明显的频次相关性
