# exp006: FiBiNET + AutoFIS - 特征交互与自动选择

## 对应 Paper

### FiBiNET
**FiBiNET: Combining Feature Importance and Bilinear feature Interaction NETwork for Click-Through Rate Prediction**

- **会议**: RecSys 2019
- **作者**: Tongwen Huang et al.
- **链接**: https://dl.acm.org/doi/10.1145/3298689.3347043

### AutoFIS
**AutoFIS: Automatic Feature Interaction Selection in Recommender Systems**

- **会议**: KDD 2020
- **作者**: Wenqi Liu et al.
- **链接**: https://dl.acm.org/doi/10.1145/3394486.3403235

## 核心思想

### FiBiNET: 特征重要性 + 双线性特征交互

#### 问题动机

传统 CTR 模型的特征交互问题：
- **FM/DeepFM**: 所有特征对用相同的内积，表达能力有限
- **注意力机制**: 计算复杂度高
- **特征重要性被忽略**: 不同特征对预测的贡献不同

#### 方法概述

FiBiNET = **Feature Importance** + **Bilinear Interaction**

```
┌─────────────────────────────────────┐
│  Input Features (Embeddings)        │
│  [e₁, e₂, ..., eₙ]                  │
└─────────────────────────────────────┘
         ↓                    ↓
┌─────────────────┐  ┌─────────────────┐
│ SENET Layer     │  │ Bilinear Layer  │ ← 双线性交互
│ (特征重要性加权) │  │                 │
└─────────────────┘  └─────────────────┘
         ↓                    ↓
         └────────┬───────────┘
                  ↓
         ┌─────────────────┐
         │ Concat + DNN    │
         └─────────────────┘
                  ↓
         ┌─────────────────┐
         │ Output (CTR)    │
         └─────────────────┘
```

#### SENET 特征重要性层

用注意力机制学习每个特征的重要性权重：

```python
# 全局池化
z = mean(embeddings, dim=1)  # [batch, embed_dim]

# 注意力权重
a = ReLU(W₁ * z)             # [batch, hidden]
s = sigmoid(W₂ * a)          # [batch, n_features]

# 加权
weighted_embeddings = s.unsqueeze(-1) * embeddings
```

#### 双线性特征交互

用可学习的矩阵增强特征交互：

```python
# 传统内积：<eᵢ, eⱼ>
interaction_ij = dot(e_i, e_j)

# 双线性：<eᵢ, W*eⱼ>
interaction_ij = dot(e_i, W @ e_j)
```

W 是 learnable 参数，可以捕获更复杂的交互模式。

---

### AutoFIS: 自动化特征交互选择

#### 问题动机

不是所有特征交互都有用：
- 某些交互是冗余的（如 city × country）
- 某些交互是噪声的（稀疏特征×稀疏特征）
- 人工筛选成本高

#### 方法概述

AutoFIS 为**每对特征交互学习一个权重**，自动筛选有用的交互：

```python
# 为每对 (i, j) 学习权重 αᵢⱼ
interaction = Σᵢⱼ αᵢⱼ * dot(eᵢ, eⱼ)

# αᵢⱼ 通过 Gumbel-Softmax 或连续松弛学习
αᵢⱼ = softmax(wᵢⱼ / τ)  # τ 是温度参数
```

训练后，αᵢⱼ ≈ 0 的交互可以剪枝。

#### 两阶段训练

1. **搜索阶段**: 联合优化模型参数 + 交互权重 α
2. **重训练阶段**: 只保留 top-k 交互，重新训练

## 本项目实现

### FiBiNET 简化版

```python
class FiBiNET(nn.Module):
    def __init__(self, n_features, embed_dim):
        # SENET 降维
        self.se_reduction = Linear(n_features, embed_dim // 2)
        self.se_expand = Linear(embed_dim // 2, n_features)
        
        # 双线性交互
        self.bilinear = Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, embeddings):
        # SENET
        z = embeddings.mean(dim=1)
        a = relu(self.se_reduction(z))
        s = sigmoid(self.se_expand(a))
        weighted = s.unsqueeze(-1) * embeddings
        
        # 双线性交互
        bilinear_emb = self.bilinear(embeddings)
        interactions = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                interactions.append(weighted[:,i] * bilinear_emb[:,j])
        
        return concat(interactions)
```

### AutoFIS 简化版

```python
# 为每对特征学习权重
self.interaction_weights = Parameter(torch.randn(n_pairs))

# 前向传播
interactions = []
for idx, (i, j) in enumerate(feature_pairs):
    alpha = sigmoid(self.interaction_weights[idx])
    interaction = alpha * dot(e_i, e_j)
    interactions.append(interaction)
```

## 预期收益

- **FiBiNET**: 相比 DeepFM，AUC +0.5~1 千分点
- **AutoFIS**: 减少 30%~50% 交互，AUC 持平
- **可解释性**: 重要交互权重可用于分析

## 实验设计

### 对比方案
1. **DeepFM**: 标准基线
2. **FiBiNET**: SENET + 双线性
3. **AutoFIS**: 带交互权重选择
4. **FiBiNET+AutoFIS**: 结合两者

### 评估指标
- AUC / PCOC
- 有效交互数量（AutoFIS）
- 特征重要性排序（SENET）

## 参考实现

- 本项目实现：`src/models/fibinet.py`（待创建）
