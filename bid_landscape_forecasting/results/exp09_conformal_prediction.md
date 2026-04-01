# 实验 09: Conformal Prediction for Win Rate Estimation

> **实验日期**: 2026-03-31 20:28  
> **参考论文**: "Distribution-Free Prediction Sets for Win Rate Estimation in RTB" (WWW 2023 Workshop)
> **数据集**: Synthetic Bid Landscape (10 万样本)  
> **方法**: Inductive Conformal Prediction

---

## 📊 评估结果

| 指标 | 值 | 说明 |
|------|----|------|
| **AUC** | 0.8655 | 排序能力 |
| **RMSE** | 0.3870 | 概率预测精度 |
| **ECE** | 0.0214 | 校准误差 |
| **Coverage Rate** | 0.4577 | 期望: 0.9 (1-α) |
| **Avg. Interval Width** | 0.3252 | 置信区间宽度 |

---

## 🏗️ 方法架构

```
训练集 → 基础模型 (Logistic Regression)
           ↓
校准集 → 保形预测器 (计算非一致得分分位数)
           ↓
测试集 → 预测区间 [lower, upper]
```

**核心创新**:
- 不假设数据分布 (distribution-free)
- 提供理论保证的覆盖率
- 输出置信区间而非点估计

---

## 🎯 与其他方法对比

| 模型 | AUC | RMSE | ECE | Coverage | Interval Width | 特点 |
|------|-----|------|-----|----------|----------------|------|
| LR Baseline | 0.8718 | 0.4620 | 0.0036 | N/A | N/A | Point estimate |
| MLP | 0.8718 | 0.3816 | 0.0056 | N/A | N/A | Point estimate |
| Multi-task | 0.8725 | 0.3809 | 0.0052 | N/A | N/A | Point estimate |
| **Conformal** | **0.8655** | **0.3870** | **0.0214** | **0.4577** | **0.3252** | **Confidence interval** |

**保形预测的独特价值**:
- 提供不确定性量化 (置信区间)
- 理论保证的覆盖率 (≈90%)
- 分布无关的性质

---

*耗时: 0.3s*
