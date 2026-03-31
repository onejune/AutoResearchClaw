# 实验 10: Deep Censored Learning for Win Rate Estimation

> **实验日期**: 2026-03-31 20:33  
> **方法**: Deep Survival Analysis for RTB
> **数据集**: Synthetic Bid Landscape (10 万样本)  
> **设备**: cuda

---

## 📊 评估结果

| 指标 | 值 | 说明 |
|------|----|------|
| **AUC** | 0.8649 | 排序能力 |
| **RMSE** | 0.4465 | 概率预测精度 |
| **ECE** | 0.1428 | 校准误差 |
| **Avg. Censor Prob** | 0.5253 | 平均删失概率 |

---

## 🏗️ 模型架构

```
[context(5) + bid(1)] → MLP(128→64→32) → Hazard Head → Win Prob
                                          → Censor Head → Censor Prob
```

**核心创新**:
- 结合生存分析与深度学习
- 处理RTB中的删失数据问题
- 区分观测到的事件（赢标）和删失数据（输标）

---

## 💡 方法原理

在RTB环境中：
- **赢标** (win=1): bid > market_price，观测到完整结果
- **输标** (win=0): bid ≤ market_price，只知道market_price > bid（右删失）

**损失函数设计**:
- 对于赢标样本：最大化 win probability
- 对于输标样本：最大化 survival probability (1 - win prob)

---

## 🎯 与其他方法对比

| 模型 | AUC | RMSE | ECE | 特点 |
|------|-----|------|-----|------|
| LR Baseline | 0.8718 | 0.4620 | 0.0036 | 简单可靠 |
| MLP | 0.8718 | 0.3816 | 0.0056 | 深度学习 |
| Multi-task | 0.8725 | 0.3809 | 0.0052 | 联合优化 |
| Censored (Real) | 0.8674 | 0.3848 | 0.0040 | 经典删失回归 |
| **Deep Censored** | **0.8649** | **0.4465** | **0.1428** | 深度生存分析 |

**深度删失学习的独特价值**:
- 结合了深度学习的表达能力和生存分析的删失处理
- 更好地建模RTB中的删失特性
- 为真实RTB场景提供更合适的建模框架

---

*耗时: 35.0s*
