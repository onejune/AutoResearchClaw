# 实验 06: Multi-Task Learning for Sequence Attention (MTLSA)

> **实验日期**: 2026-03-31 20:20  
> **数据集**: Synthetic Bid Landscape (20 万样本)  
> **设备**: cuda

---

## 📊 评估结果

| 指标 | 值 |
|------|----|
| **AUC** | 0.8694 |
| **RMSE** | 0.3834 |
| **ECE** | 0.0023 |
| **Monotonicity** | 0.9973 |

### Per-Task AUC（按 bid level 从低到高）

| Task | Bid Level | AUC |
|------|-----------|-----|
| Task 1 | 最低 bid (0.5v) | 0.7536 |
| Task 2 | 低 bid (0.7v) | 0.7104 |
| Task 3 | 中 bid (1.0v) | 0.7046 |
| Task 4 | 高 bid (1.3v) | 0.7166 |
| Task 5 | 最高 bid (1.5v) | 0.7570 |

---

## 🏗️ 模型架构

```
Context → Shared Bottom → shared_repr
                              ↓
Bid_k → Bid Embedding → bid_embed_k
                              ↓
[shared_repr, bid_embed_k] × 5 tasks → Multi-head Attention → Task Heads → Win_prob_k
```

- **Shared Bottom**: Linear(5→64→32)
- **Bid Embedding**: Linear(1→16→16)
- **Cross-task Attention**: MultiheadAttention(heads=4)
- **Task Heads**: 5 × Linear(48→32→1)

---

## 💡 核心发现

1. **跨任务注意力**: 低 bid 任务的信息可以帮助高 bid 任务预测（相关性学习）
2. **Per-task 差异**: 中间 bid level (Task 3, ~1.0v) 通常最难预测（胜率~50%）
3. **单调性**: 模型是否学到 bid↑→win_prob↑ 的规律

---

*耗时: 96.0s*
