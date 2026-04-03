# exp002: Data Distribution Search (DDS) - 数据分布感知的维度搜索

## 对应 Paper

**Data Distribution Search for Automated Embedding Dimensionality in Recommender Systems**

- **会议**: WWW 2022 (或类似期刊/会议)
- **主题**: 基于数据分布特性的自动化 embedding 维度分配

## 核心思想

### 问题动机

AutoEmb 等方法的局限：
- 仅依赖**频次统计**（如 vocab size、样本数）
- 忽略了**数据分布特性**（如偏度、峰度、稀疏模式）

不同特征的数据分布差异很大：
- **长尾分布**：少数 ID 占大部分样本（如 adid）
- **均匀分布**：ID 样本相对均衡（如 gender）
- **多峰分布**：多个群体各有集中（如 city）

### 方法概述

通过分析特征的**数据分布统计量**，自动推荐最优 embedding 维度：

```python
分布统计量 = {
    "skewness": 偏度（长尾程度）,
    "kurtosis": 峰度（集中程度）,
    "gini_coefficient": 基尼系数（不平等程度）,
    "coverage@k": top-k 覆盖率
}

推荐维度 = f(分布统计量，vocab_size)
```

### 关键分布指标

| 指标 | 公式/说明 | 推荐维度 |
|------|-----------|----------|
| **偏度 (Skewness)** | 分布对称性，>1 表示长尾 | 高偏度→低维（避免过拟合尾部） |
| **峰度 (Kurtosis)** | 分布尖锐程度 | 高峰度→低维（集中在少数 ID） |
| **基尼系数** | 0=完全平等，1=完全不平等 | 高基尼→低维 |
| **Coverage@10** | top-10 ID 占比 | 高覆盖→低维 |

### 决策规则示例

```python
if skewness > 2.0 and gini > 0.8:
    dim = 16  # 极端长尾，低维防止过拟合
elif skewness < 0.5 and gini < 0.3:
    dim = 64  # 均匀分布，高维捕获细节
else:
    dim = 32  # 中等
```

## 与 AutoEmb 的区别

| 方面 | AutoEmb | DDS |
|------|---------|-----|
| **输入信号** | vocab size, 频次 | 偏度、峰度、基尼系数等 |
| **搜索方法** | RL 或启发式 | 基于分布统计的规则 |
| **优势** | 端到端优化 | 可解释性强，无需训练 |
| **劣势** | 搜索成本高 | 规则需要调优 |

## 本项目实现

简化版 DDS，使用以下规则：

```python
def suggest_dim_by_distribution(freqs):
    skewness = compute_skewness(freqs)
    gini = compute_gini(freqs)
    
    if skewness > 3.0 or gini > 0.9:
        return 16  # 极端长尾
    elif skewness > 1.5 or gini > 0.7:
        return 32  # 长尾
    else:
        return 64  # 相对均匀
```

## 预期收益

- 相比纯频次方法，更精准识别**易过拟合特征**
- 参数量进一步减少 10%~20%
- 尾部 ID 泛化能力提升

## 实验设计

### 对比方案
1. **Fixed-64**: 所有特征 64 维
2. **AutoEmb**: 基于频次的启发式
3. **DDS**: 基于分布统计

### 评估指标
- AUC / PCOC
- 参数量
- 各特征维度分配对比

## 参考实现

- 本项目实现：`src/methods/dimension_search.py`（待创建）
