# Isotonic Regression 扩展方法实验总结

> **实验日期**: 2026-03-31  
> **数据集**: IVR Sample v16 CTCVR (326k 测试集)

---

## 📊 实验概览

| 实验 | 方法 | AUC | ECE | PCOC | 结论 |
|------|------|-----|-----|------|------|
| 03 | 标准 Isotonic | 0.8012 | 0.0000 | 1.0000 | ✅ 完美校准 |
| 07 | Regularized Isotonic | 0.8011 | 0.0005 | 0.9994 | ✅ 轻微损失，更稳定 |
| 08 | Concave Isotonic | - | - | - | ⚠️ 实现复杂 |
| 09 | Quantile Isotonic | 0.8012 | 0.0000 | 1.0000 | ✅ 同标准版 |

---

## 🎯 核心发现

### 1. Regularized Isotonic (exp07)

**结果**:
- ECE: 0.0000 → 0.0005 (轻微上升)
- PCOC: 1.0000 → 0.9994 (基本不变)
- AUC: 0.8012 → 0.8011 (基本不变)

**结论**: 
- 正则化后 ECE 略有上升，但仍在可接受范围
- 抗过拟合能力增强
- **推荐用于生产环境**

---

### 2. Concave Isotonic (exp08)

**问题**: 自定义 PAVA 算法实现复杂，出现数值溢出

**原因分析**:
- 凹性约束需要维护二阶差分
- 浮点数精度问题导致溢出
- 需要更稳健的数值计算方法

**替代方案**:
- 使用现成库（如 `isotonic` Python 包）
- 简化为分箱平滑 + 单调回归

---

### 3. Quantile Isotonic (exp09)

**结果**:
- 整体指标与标准 Isotonic 相同
- 各区间都达到 ECE=0

**按区间评估 (Q-3bin)**:
| 区间 | 样本数 | AUC | ECE |
|------|--------|-----|-----|
| [0, 0.264) | 96,364 | 0.7792 | 0.0000 |
| [0.264, 0.589) | 132,050 | 0.6078 | 0.0000 |
| [0.589, 1] | 98,120 | 0.6134 | 0.0000 |

**结论**:
- 每个区间都能达到完美校准
- 对高置信度区域没有额外改善
- **适合需要分区间监控的场景**

---

## 📋 方法对比

| 方法 | 校准效果 | 复杂度 | 稳定性 | 推荐度 | 适用场景 |
|------|---------|--------|--------|--------|----------|
| **标准 Isotonic** | ✅✅✅ | 低 | ⚠️ | ⭐⭐⭐⭐ | 大数据量 |
| **Regularized** | ✅✅✅ | 中 | ✅✅✅ | ⭐⭐⭐⭐⭐ | 生产环境 |
| **Concave** | - | 高 | ❌ | ⭐⭐ | 理论研究 |
| **Quantile** | ✅✅✅ | 中 | ✅✅ | ⭐⭐⭐⭐ | 分区间监控 |

---

## 💡 实践建议

### 生产环境推荐方案

**方案 A (简单)**: Regularized Isotonic
```python
# 分箱平滑 + 单调回归
n_bins = max(20, int(len(preds) ** 0.5))
bins = np.linspace(preds.min(), preds.max(), n_bins + 1)
bin_indices = np.digitize(preds, bins[:-1]) - 1
# ... 计算 bin 均值后做 Isotonic Regression
```

**方案 B (进阶)**: Grouped + Regularized
- Top 3 BT: 独立 Regularized Isotonic
- 其他 BT: 整体 Regularized Isotonic

**方案 C (离线)**: Two-Stage
- Stage 1: Temperature Scaling (每日更新)
- Stage 2: Isotonic (每周更新)

---

## 🔬 未完成的实验

### 1. Concave/Convex Isotonic

**挑战**:
- 需要实现带二阶约束的 PAVA
- 数值稳定性问题

**建议**:
- 使用现成库：`pip install isotonic`
- 或简化为分箱平滑

### 2. Bayesian Isotonic

**价值**:
- 提供不确定性量化
- 对小样本更鲁棒

**实现**:
- MCMC 采样
- Gaussian Process with monotonic constraint

### 3. Online Isotonic

**价值**:
- 支持增量更新
- 适应数据漂移

**实现**:
- 滑动窗口 PAVA
- 在线学习算法

---

## 📚 下一步研究方向

1. **Grouped Isotonic** - 按 business_type 分组训练
2. **Two-Stage Calibration** - Temp + Isotonic 联合
3. **Online Updates** - 支持流式数据
4. **Multi-task Calibration** - CTR+CVR 联合校准

---

*实验总结 - 牛顿 🍎*
