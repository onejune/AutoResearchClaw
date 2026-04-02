# exp004: MetaEmb - 元学习 Embedding 泛化

## 对应 Paper

**MetaEmb: Learning Meta Embedding for Cold-Start Recommendation via Meta-Learning**

- **会议**: WSDM 2022 (或类似)
- **主题**: 用元学习解决新 ID 冷启动问题

## 核心思想

### 问题动机

传统 embedding 的冷启动问题：
- 新广告/新物品上线时，没有历史交互
- 无法更新 embedding（梯度为 0）
- 只能用随机初始化或全局平均，效果差

### 方法概述

MetaEmb 用**元学习**训练一个**embedding 生成器**：

```
┌──────────────────────────────────────┐
│  新 ID 的侧信息 (Side Information)    │
│  - 文本描述 (title, category)         │
│  - 图像特征 (如有)                    │
│  - 创建者信息 (advertiser)            │
└──────────────────────────────────────┘
                ↓
     ┌─────────────────────┐
     │  Meta Learner       │ ← 元学习训练
     │  (MLP / Transformer) │
     └─────────────────────┘
                ↓
     ┌─────────────────────┐
     │  生成的 Embedding    │ ← 直接用于预测
     │  [batch, dim]       │
     └─────────────────────┘
```

### 关键技术

#### 1. 侧信息编码

将新 ID 的辅助信息编码为向量：

```python
# 文本侧信息（如 category 名称）
category_emb = TextEncoder("Electronics > Phones")

# 结构化侧信息（如 advertiser_id, create_time）
side_emb = concat(
    Embedding(advertiser_id),
    TimeEmbedding(create_time)
)
```

#### 2. Meta Learner

学习从侧信息到 embedding 的映射：

```python
class MetaLearner(nn.Module):
    def __init__(self, side_info_dim, embedding_dim):
        self.generator = nn.Sequential(
            nn.Linear(side_info_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, side_info):
        return self.generator(side_info)
```

#### 3. 元学习训练策略

**Episodic Training**（ episod 训练）：

1. 构造"伪冷启动"任务：随机隐藏部分 ID 的 embedding
2. 用侧信息生成 embedding
3. 在预测任务上计算 loss
4. 反向传播更新 Meta Learner

```python
for episode in episodes:
    # 随机选一批 ID 作为"新 ID"
    cold_ids = sample_ids()
    
    # 移除这些 ID 的 embedding（模拟冷启动）
    remove_embeddings(cold_ids)
    
    # 用侧信息生成 embedding
    generated_emb = meta_learner(get_side_info(cold_ids))
    
    # 计算预测 loss
    loss = criterion(model(x), y)
    
    # 更新 meta learner
    loss.backward()
```

### 本项目简化实现

由于完整元学习训练复杂，本项目采用**简化版**：

```python
# 用已有特征组合作为侧信息
side_info = concat(
    Embedding(business_type),
    Embedding(advertiser),
    Embedding(category)
)

# 直接用 MLP 生成 pseudo embedding
generated_emb = MLP(side_info)
```

## 预期收益

- **冷启动 ID 效果提升**：新广告 CTR 预估更准
- **泛化能力增强**：相似 ID 的 embedding 接近
- **可迁移性**：跨域场景下可复用

## 实验设计

### 冷启动模拟实验

1. 训练时隐藏部分 ID（从训练集中移除）
2. 测试时只用这些"新 ID"
3. 对比：
   - Random Init（随机初始化）
   - Global Average（全局平均）
   - MetaEmb（生成式）

### 正常场景实验

- 所有 ID 正常训练
- 对比 MetaEmb 与普通 embedding 的 AUC

## 参考实现

- 本项目实现：`src/methods/meta_learner.py`（待创建）
