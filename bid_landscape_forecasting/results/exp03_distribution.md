# 实验 03: Distribution Modeling for Bid Landscape Forecasting

> **实验日期**: 2026-03-31 19:30  
> **数据集**: Synthetic Bid Landscape (50 万样本)  
> **设备**: CUDA GPU

---

## 📊 方法对比

| Method | AUC | RMSE | ECE | Mean Variance |
|--------|-----|------|-----|---------------|
| Logistic Regression | 0.8718 | 0.4620 | 0.0036 | N/A |
| MLP | 0.8718 | 0.3816 | 0.0056 | N/A |
| **Beta Regression** | **0.8718** | **0.4087** | **0.1221** ⚠️ | **0.2017** |

---

## 🎯 核心发现

### 1. AUC 持平，但 ECE 显著恶化
- **AUC**: 0.8718 (与 LR/MLP 完全相同)
- **ECE**: 0.1221 (比 LR 高 33 倍！⚠️)
- **原因**: Beta distribution 的 uncertainty weighting 破坏了校准

### 2. 不确定性建模成功
- **Mean Variance**: 0.2017 (合理的方差范围)
- **说明**: 模型确实学到了不同样本的不确定性
- **问题**: 这种不确定性没有被正确用于校准

### 3. RMSE 介于 LR 和 MLP 之间
- **Beta Reg RMSE**: 0.4087
- **LR RMSE**: 0.4620
- **MLP RMSE**: 0.3816
- **结论**: Uncertainty-aware training 有一定帮助，但不如纯 MLP

---

## 🔍 关键洞察

### Beta Regression 的优势
1. **Uncertainty Quantification**: 可以输出每个预测的置信度
2. **Risk-sensitive Applications**: 对于高风险决策场景有价值
3. **Active Learning**: 可以用高不确定性样本指导数据标注

### Beta Regression 的劣势
1. **Calibration 恶化**: ECE 从 0.0036 → 0.1221
2. **训练不稳定**: Loss 波动大 (2.3 vs 0.45 BCE)
3. **计算开销**: 需要输出两个参数 (α, β)

### 适用场景建议
| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| **生产环境 (追求校准)** | Logistic Regression | ECE < 0.01 |
| **概率精度要求高** | MLP | RMSE 最低 |
| **需要不确定性估计** | Beta Regression | 唯一能提供 variance |
| **高风险决策** | Beta Regression + Calibration | 先用 Beta Reg，再用 Isotonic 校准 |

---

## 💡 下一步改进方向

### 短期
1. **Post-hoc Calibration**: Beta Regression + Isotonic Regression
2. **Loss Function Tuning**: 调整 uncertainty weighting 策略
3. **Ensemble Methods**: 结合多个模型的 uncertainty

### 中期
1. **Mixture Density Networks**: 更灵活的分布建模
2. **Bayesian Neural Networks**: 真正的贝叶斯不确定性
3. **Conformal Prediction**: 理论保证的置信区间

---

*实验报告 - 牛顿 🍎*
