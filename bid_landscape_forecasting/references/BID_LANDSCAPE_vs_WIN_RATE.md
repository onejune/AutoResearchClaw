# Bid Landscape vs Win Rate Estimation - 概念辨析

> **创建时间**: 2026-03-31 19:53  
> **目的**: 澄清两个容易混淆的概念

---

## 📚 定义对比

### 1. Win Rate Estimation (胜率预估)

**问题**: 给定一个 bid，预测赢标的概率

```
P(win | bid, context) = ?
```

**输入**:
- bid amount (出价)
- context features (广告特征、用户特征、场景特征等)

**输出**:
- win probability (0~1 之间的概率值)

**典型应用**:
- Real-time bidding (RTB) 中的出价决策
- Budget pacing (预算控制)
- ROI optimization

---

### 2. Bid Landscape Forecasting (竞价景观预测)

**问题**: 预测整个 bid-win 关系曲线

```
f(bid) = P(win | bid, context), for bid ∈ [min_bid, max_bid]
```

**输入**:
- context features (广告特征、用户特征、场景特征等)
- **bid range** (一系列可能的出价)

**输出**:
- **完整的 bid-win 曲线** (函数或离散点集)
- 或者曲线的参数化表示 (如 sigmoid 的参数)

**典型应用**:
- Optimal bid search (寻找最优出价)
- Bid strategy learning (学习出价策略)
- Auction simulation (拍卖模拟)

---

## 🔑 核心区别

| 维度 | Win Rate Estimation | Bid Landscape Forecasting |
|------|---------------------|---------------------------|
| **输出** | 单点概率 | 完整曲线/函数 |
| **粒度** | Point-wise | Function-level |
| **用途** | 评估特定 bid 的胜率 | 探索最优 bid |
| **模型** | Classifier (LR, XGB, NN) | Regression / Distribution modeling |
| **数据需求** | (bid, win) pairs | (context, bid, win) triples |

---

## 🎯 数学关系

### Win Rate Estimation
```python
# 单点预测
model = train_classifier(features + bid_amount, win_label)
p_win = model.predict(context_features, specific_bid)
```

### Bid Landscape Forecasting
```python
# 曲线预测
model = train_landscape_model(context_features, (bid, win) pairs)

# 方法 1: 直接输出曲线参数
alpha, beta = model.predict(context_features)
def win_prob(bid):
    return sigmoid(alpha * bid + beta)

# 方法 2: 条件概率建模
def win_prob(bid):
    return model.predict(context_features, bid)

# 方法 3: 生成多个 bid 点
bids = np.linspace(min_bid, max_bid, 100)
probs = [model.predict(context_features, b) for b in bids]
```

---

## 💡 我们的项目定位

### 当前实验：Bid Landscape Forecasting ✅

**为什么是 Bid Landscape?**

1. **数据生成方式**: 
   ```python
   # 我们为每个样本生成了 5 个 bid levels
   bid_levels = {0.5v, 0.7v, v, 1.3v, 1.5v}
   # 这天然形成了 bid-win 曲线上的 5 个点
   ```

2. **模型设计**:
   - Multi-task: 同时学习 CTR 和 Win
   - Shared representation: 学习 context 到 bid-win 曲线的映射
   - Beta Regression: 输出分布参数而非单点

3. **应用场景**:
   - 找到最优 bid (通过搜索曲线)
   - 理解 bid sensitivity (曲线斜率)
   - Budget allocation across auctions

### 如果做 Win Rate Estimation 会怎样？

**简化版问题**:
```python
# 只预测特定 bid 的胜率
model = LogisticRegression()
model.fit([context_features, bid_amount], win_label)
p_win = model.predict(test_context, test_bid)
```

**局限性**:
- ❌ 无法直接得到最优 bid
- ❌ 需要多次推理才能画出曲线
- ❌ 忽略了 bid 之间的相关性

---

## 🔄 转换关系

### Bid Landscape → Win Rate Estimation

```python
# 从 landscape 中提取单点胜率
landscape = predict_landscape(context_features)  # Returns function f(bid)
p_win_at_10 = landscape(10.0)  # Query at specific bid
```

**优势**:
- ✅ 一次推理，任意查询
- ✅ 可以利用曲线平滑性
- ✅ 可以求导找最优 bid

### Win Rate Estimation → Bid Landscape

```python
# 通过多次查询构建曲线
def build_landscape(context, bid_range):
    curve = []
    for bid in bid_range:
        p_win = win_rate_model(context, bid)
        curve.append((bid, p_win))
    return curve
```

**劣势**:
- ❌ 需要 N 次推理 (N = bid_range 长度)
- ❌ 无法保证曲线平滑
- ❌ 计算成本高

---

## 📊 实际案例对比

### 场景：RTB 系统中的出价决策

#### Win Rate Estimation 方案
```python
# 固定几个候选 bid，选最好的
candidate_bids = [5.0, 10.0, 15.0, 20.0, 25.0]
best_bid = None
best_value = -inf

for bid in candidate_bids:
    p_win = win_rate_model(context, bid)
    value = p_win * value_per_conversion - bid
    if value > best_value:
        best_value = value
        best_bid = bid

return best_bid  # 只能从候选集中选
```

#### Bid Landscape Forecasting 方案
```python
# 直接优化连续函数
landscape = predict_landscape(context)  # Returns smooth function

def objective(bid):
    p_win = landscape(bid)
    return p_win * value_per_conversion - bid

# 使用梯度下降或其他优化方法
optimal_bid = optimize(objective, bounds=[min_bid, max_bid])

return optimal_bid  # 可以是任意实数
```

**优势**:
- ✅ 更精细的 bid 控制
- ✅ 可以利用梯度信息
- ✅ 计算效率更高 (一次推理 vs 多次)

---

## 🎓 学术文献中的命名

### 常见术语

| 术语 | 英文 | 含义 |
|------|------|------|
| **Bid Landscape** | Bid Landscape / Win Probability Curve | 完整的 bid-win 关系 |
| **Win Rate** | Win Rate / Win Probability | 单点胜率 |
| **pCTR** | predicted Click-Through Rate | 点击率预估 |
| **pCVR** | predicted Conversion Rate | 转化率预估 |
| **pCTR/pWin** | predicted Win Rate | 胜率预估 |

### 代表性论文

1. **"Predicting the Probability of Winning in Real-Time Bidding"** (KDD 2017)
   - 标题用 "Probability of Winning"
   - 实际做的是 Bid Landscape (预测不同 bid 的胜率)

2. **"Bid Landscape Estimation with Deep Learning"** (WWW 2020)
   - 明确使用 "Bid Landscape"
   - 输出完整的 bid-win 曲线

3. **"Deep Bidding: Learning to Bid with Deep Neural Networks"** (KDD 2019)
   - End-to-end 学习最优 bid
   - 隐式学习了 bid landscape

---

## 💡 总结

### 本质关系

**Bid Landscape Forecasting ⊃ Win Rate Estimation**

- **Win Rate Estimation** 是 Bid Landscape 在特定点的查询
- **Bid Landscape Forecasting** 是学习整个函数关系

### 我们的项目

**定位**: Bid Landscape Forecasting

**理由**:
1. 数据生成方式天然支持 (5 bid levels per sample)
2. 模型设计面向曲线学习 (shared representation)
3. 应用场景需要最优 bid 搜索

**价值**:
- ✅ 比 Win Rate Estimation 更通用
- ✅ 支持连续优化
- ✅ 计算效率更高
- ✅ 更符合工业界需求

---

*概念辨析 - 牛顿 🍎*
