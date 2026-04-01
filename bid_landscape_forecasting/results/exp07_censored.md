# 实验 07: Censored Regression (Tobit Model)

> **实验日期**: 2026-03-31 20:28  
> **数据集**: Synthetic Bid Landscape (20 万样本)  
> **设备**: cuda

---

## 📊 评估结果

| 模式 | AUC | RMSE | ECE | 说明 |
|------|-----|------|-----|------|
| **Censored** | 0.8700 | 0.3830 | 0.0066 | 真实场景（不用 true_value） |
| **Oracle** | 0.8663 | 0.3857 | 0.0027 | 使用 true_value（上界） |

---

## 🏗️ 模型架构

```
[context(5) + bid(1)] → MLP(128→64) → Win Head (sigmoid) → P(win)
                                     → Price Head (softplus) → market_price_est
```

## 💡 损失函数

```
L = L_uncensored + L_censored [+ α × L_price]

L_uncensored = -log P(win=1)   (赢标样本)
L_censored   = -log P(win=0)   (输标样本，只知道输了)
L_price      = MSE(price_est, true_value)  (仅 oracle 模式)
```

---

## 🎯 核心发现

1. **Censored vs Oracle 差距**: Oracle 因为有额外监督信号（true_value），预期表现更好
2. **Censored 的实际意义**: 这才是真实 RTB 场景的正确建模方式
3. **与基线对比**: 
   - LR 基线: AUC=0.8718, ECE=0.0036
   - Censored 模型是否能超越简单 LR？

---

*耗时: 242.5s*
