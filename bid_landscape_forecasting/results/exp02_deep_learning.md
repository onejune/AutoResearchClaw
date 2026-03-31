# 实验 02: Deep Learning Models

> **实验日期**: 2026-03-31 19:14  
> **数据集**: Synthetic Bid Landscape (50 万样本)  
> **设备**: CUDA GPU

---

## 📊 方法对比

| Method | AUC | RMSE | ECE |
|--------|-----|------|-----|
| **MLP** (256→128→64) | **0.8718** | **0.3816** | **0.0056** |

---

## 🎯 与基线对比

| Method | AUC | RMSE | ECE |
|--------|-----|------|-----|
| Logistic Regression (exp01) | 0.8718 | 0.4620 | 0.0036 |
| XGBoost (exp01) | 0.8714 | 0.4625 | 0.0050 |
| **MLP (exp02)** | **0.8718** | **0.3816** | **0.0056** |

---

## 💡 核心发现

### 1. MLP vs LR/XGB: AUC 持平
- **AUC**: 0.8718 (与 LR 完全相同)
- **提升**: +0.0000 vs LR, +0.0004 vs XGB
- **结论**: Deep learning 在合成数据上**没有优势**

### 2. RMSE 显著改善
- **MLP RMSE**: 0.3816
- **LR RMSE**: 0.4620
- **改进**: -17.4% (概率预测更准确)
- **原因**: MLP 能更好地拟合 sigmoid 曲线

### 3. 校准效果略差
- **MLP ECE**: 0.0056
- **LR ECE**: 0.0036
- **差异**: +56% (但仍 < 0.01，非常理想)
- **可能原因**: Dropout 引入随机性

### 4. 训练收敛快速
```
Epoch 10: Val AUC=0.8717
Epoch 50: Val AUC=0.8718
```
- 10 个 epoch 就达到最优
- 后续几乎没有提升 (过拟合风险低)

---

## 🔍 关键洞察

1. **简单模型已足够**: LR/XGB 已经达到理论上限
2. **Deep learning 的优势不在 AUC**: 
   - RMSE 改善 17% → 概率预测更准
   - 但分类边界已经清晰，AUC 无法提升
3. **合成数据的局限性**:
   - Sigmoid 假设太强
   - 真实 bid landscape 可能有更复杂的非线性关系
4. **下一步方向**:
   - 尝试更复杂的数据生成策略
   - 或转向真实数据验证

---

## 📈 下一步实验

### exp03: Distribution Modeling
- Beta Regression (输出完整分布)
- Mixture Density Networks
- Uncertainty quantification

### exp04: Multi-task Learning
- Joint CTR + Bid Landscape
- Shared representation

---

*实验报告 - 牛顿 🍎*
