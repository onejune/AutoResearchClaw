# 实验 05: Deep Landscape Forecasting (DLF)

> **实验日期**: 2026-03-31 20:20  
> **数据集**: Synthetic Bid Landscape (20 万样本)  
> **设备**: cuda

---

## 📊 评估结果

| 指标 | 值 | 说明 |
|------|----|------|
| **AUC** | 0.8687 | 胜率预测排序能力 |
| **RMSE** | 0.3841 | 概率预测精度 |
| **ECE** | 0.0052 | 校准误差 |
| **Monotonicity** | 0.9997 | bid↑→win_prob↑ 的比例 |

---

## 🏗️ 模型架构

```
Context Features → MLP Encoder → Context Vector
                                        ↓
Sorted Bid Sequence → [bid_i, ctx] → GRU → Output Head → Win Prob_i
```

- **Context Encoder**: Linear(5→64→32) + ReLU + Dropout
- **GRU**: 2 layers, hidden=32, input=[bid(1) + ctx(32)]
- **Output**: Linear(32→16→1) + Sigmoid

---

## 🎯 核心发现

### DLF vs 基线对比

| 模型 | AUC | RMSE | ECE | 特点 |
|------|-----|------|-----|------|
| Logistic Regression | 0.8718 | 0.4620 | **0.0036** | 最佳校准 |
| MLP | 0.8718 | **0.3816** | 0.0056 | 最佳 RMSE |
| Multi-task | **0.8725** | 0.3809 | 0.0052 | 最佳 AUC |
| **DLF (本实验)** | **0.8687** | **0.3841** | **0.0052** | 序列建模 |

### 单调性分析
- Monotonicity = 0.9997
- DLF 通过 RNN 建模价格序列，天然倾向于学习单调递增的 bid-win 关系
- 理想情况下 Monotonicity 应接近 1.0

---

## 💡 方法优势

1. **序列感知**: GRU 能捕捉 bid 之间的相对关系（不只是单点预测）
2. **完整曲线**: 一次前向传播输出整条 bid-win 曲线
3. **可扩展**: 可以预测任意 bid 点（不限于训练时的 5 个）

---

*耗时: 85.8s*
