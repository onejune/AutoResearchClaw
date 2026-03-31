# 实验 01: Bid Landscape Forecasting Baseline

> **实验日期**: 2026-03-31 18:52  
> **数据集**: Synthetic Bid Landscape (50 万样本)

---

## 📊 方法对比

| Method | AUC | RMSE | MAE | ECE |
|--------|-----|------|-----|-----|
| **Logistic Regression** | **0.8718** | **0.4620** | **0.2135** | **0.0036** |
| XGBoost | 0.8714 | 0.4625 | 0.2139 | 0.0050 |
| Quantile Regression (median) | 0.5000 | 0.7060 | N/A | N/A |

---

## 🎯 核心发现

### 1. Logistic Regression 最佳
- **AUC**: 0.8718 (略优于 XGBoost)
- **ECE**: 0.0036 (校准非常好)
- **原因**: 合成数据本身就是 sigmoid 分布，LR 完美匹配

### 2. XGBoost 表现相当
- **AUC**: 0.8714 (与 LR 几乎相同)
- **特征重要性**:
  - `bid_amount`: 92.95% (主导特征)
  - `true_value`: 6.56%
  - `business_type`: 0.48%

### 3. Quantile Regression 失败
- **AUC**: 0.5000 (随机猜测)
- **原因**: Quantile Regression 输出连续值，不适合二分类任务
- **改进方向**: 需要调整 loss function 或 post-processing

### 4. 校准效果优秀
- **LR ECE**: 0.0036 (< 0.01，非常理想)
- **XGB ECE**: 0.0050 (< 0.01，非常理想)
- 说明合成数据的 sigmoid 假设合理

---

## 📈 下一步实验

### exp02: Deep Learning Models
- MLP (Multi-Layer Perceptron)
- TabNet (interpretable tabular DL)
- FT-Transformer

### exp03: Distribution Modeling
- Beta Regression (output α, β)
- Mixture Density Networks
- Calibration methods (Isotonic, Temperature Scaling)

### exp04: Multi-task Learning
- Joint CTR + Bid Landscape
- Shared representation learning

---

## 💡 关键洞察

1. **合成数据质量高**: LR/XGB 都能达到 AUC > 0.87，说明 bid landscape 生成策略合理
2. **Bid amount 是主导特征**: 占 93% 的重要性，符合预期
3. **Business type 影响小**: 只有 0.5%，可能需要更强的交互特征
4. **校准天然良好**: ECE < 0.01，不需要额外校准

---

*实验报告 - 牛顿 🍎*
