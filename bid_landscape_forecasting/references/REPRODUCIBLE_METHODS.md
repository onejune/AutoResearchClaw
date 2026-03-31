# 基于当前数据集可复现的方法

> **数据集结构**:
> - 250 万行，每个原始样本对应 5 个 bid levels
> - 每组 5 行共享相同的 `true_value`（同一竞拍上下文）
> - 特征：`bid_amount`, `true_value`, `win_label`, `click_label`, `business_type`, `deviceid`, `adid`, `campaignid`
> - **关键：我们知道 true_value（合成数据特权），真实场景中输标时不知道**

---

## ✅ 可直接复现（数据完全支持）

### 1. DLF: Deep Landscape Forecasting (KDD 2019) ✅

**为什么可以复现**:
- 每个样本有 5 个 bid levels → 天然的价格序列
- 有 win_label → 可以构造 censored/uncensored 样本
- 有 context features (business_type, deviceid, adid, campaignid)

**需要的数据格式**:
```python
# 每个原始样本 → 5 行按 bid 排序的序列
# 输标 (win_label=0) → censored: 只知道 z >= bid
# 赢标 (win_label=1) → uncensored: 知道 bid > z (z 在上一个 bid 和当前 bid 之间)
```

**实现要点**:
```python
class DLF(nn.Module):
    # 特征编码器 + GRU 建模价格序列
    # 输出每个价格区间的条件胜率
    # 用概率链式法则得到累积胜率 CDF
```

**预期难点**: 需要重新组织数据（按 true_value 分组，每组5个bid排序）

---

### 2. MTLSA: Multi-Task Survival Analysis ✅

**为什么可以复现**:
- 5 个 bid levels 天然对应 5 个任务
- 每个任务预测 P(win | bid_i)
- 共享底层表示

**实现**:
```python
class MTLSA(nn.Module):
    # Shared MLP → 5 个 task heads
    # 每个 head 预测对应 bid level 的胜率
    # Monotonicity constraint: P(win|b1) ≤ P(win|b2) ≤ ... ≤ P(win|b5)
```

**优势**: 比 DLF 简单，可以快速实现

---

### 3. Quantile Regression Network ✅

**思路**: 预测市场价的不同分位数

**为什么可以复现**:
- 我们有 true_value（合成数据特权）
- 可以直接用 true_value 作为回归目标

**实现**:
```python
# 预测 P10, P25, P50, P75, P90 分位数
# 给定 context features，预测市场价分布的分位数
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
```

**注意**: 真实场景中 true_value 不可知，这是合成数据的特权

---

### 4. Censored Regression (Tobit Model) ✅

**思路**: 直接建模截断回归

**数据构造**:
```python
# 赢标 (win=1): y = true_value (已知)
# 输标 (win=0): y >= bid_amount (截断，只知道下界)
```

**实现**:
```python
# Tobit Loss
def tobit_loss(pred, y_true, y_censor, is_censored):
    # Uncensored: MSE(pred, y_true)
    # Censored: -log P(y >= y_censor) = -log(1 - CDF(y_censor))
```

---

### 5. Isotonic Regression on Bid Curve ✅

**思路**: 对每个样本的 5 个 bid-win 点做单调回归

**实现**:
```python
from sklearn.isotonic import IsotonicRegression
# 对每组 5 个 bid-win 点，拟合单调递增曲线
# 然后用 MLP 预测 isotonic 参数
```

---

## ⚠️ 需要改造数据才能复现

### 6. Normalizing Flow ⚠️

**问题**: 需要 true_value 的完整分布，而不只是 5 个点

**改造方案**:
- 用 true_value 作为目标分布
- 条件于 context features，学习 P(z | x)

**可行性**: ✅ 因为我们有 true_value

---

### 7. CGAN/VAE ⚠️

**问题**: 需要大量 (context, market_price) 对

**改造方案**:
- 用 (context_features, true_value) 作为训练对
- 生成市场价分布

**可行性**: ✅ 因为我们有 true_value

---

## ❌ 当前数据集不支持

### 8. 真实 Censorship 处理 ❌

**问题**: 我们的合成数据中，即使输标也知道 true_value

**真实场景**:
- 输标时：只知道 z >= bid（不知道真实市场价）
- 赢标时：知道 z < bid（但不知道确切的 z）

**影响**: DLF/MTLSA 的 censorship 处理部分无法真实验证

**解决方案**: 
- 模拟 censorship：输标时**故意忽略** true_value
- 或下载真实数据集（iPinYou, YOYI）

---

## 📋 推荐实验计划

### 优先级排序

| 优先级 | 实验 | 方法 | 预期时间 | 创新价值 |
|--------|------|------|----------|----------|
| 🔴 P1 | exp05 | DLF 复现 | 2h | ⭐⭐⭐⭐⭐ |
| 🔴 P1 | exp06 | MTLSA 复现 | 1h | ⭐⭐⭐⭐ |
| 🟡 P2 | exp07 | Quantile Regression NN | 1h | ⭐⭐⭐ |
| 🟡 P2 | exp08 | Censored Regression | 1.5h | ⭐⭐⭐⭐ |
| 🟢 P3 | exp09 | Normalizing Flow | 3h | ⭐⭐⭐⭐⭐ |
| 🟢 P3 | exp10 | Multi-task + DLF | 2h | ⭐⭐⭐⭐⭐ |

### 数据改造方案（模拟真实 Censorship）

```python
def simulate_censorship(df):
    """模拟真实 RTB 的 censorship 问题"""
    df = df.copy()
    
    # 输标时，隐藏 true_value（模拟真实场景）
    df.loc[df['win_label'] == 0, 'observed_market_price'] = np.nan
    
    # 赢标时，true_value 可观测
    df.loc[df['win_label'] == 1, 'observed_market_price'] = df.loc[df['win_label'] == 1, 'true_value']
    
    # 截断信息：输标时只知道 z >= bid_amount
    df['censored'] = (df['win_label'] == 0).astype(int)
    df['lower_bound'] = df['bid_amount']  # z >= lower_bound when censored
    
    return df
```

---

*可行性分析 - 牛顿 🍎*
