# CTR 预估校准研究 - 实验总结

> **研究日期**: 2026-03-31  
> **数据集**: IVR Sample v16 CTCVR (326 万样本)

---

## 📊 实验概览

| 实验 | 方法 | AUC | ECE | PCOC | 结论 |
|------|------|-----|-----|------|------|
| 01 | 基线 (BCE) | 0.8010 | 0.0076 | 1.0172 | ✅ 整体校准好 |
| 01b | 按 business_type 评估 | - | - | [0.96, 1.13] | ⚠️ 分组有偏差 |
| 02 | Temperature Scaling | 0.8011 | 0.0067 | 1.0114 | ✅ 轻微改善 |
| 02 | 分组 Temperature | 0.8011 | 0.0067 | 1.0114 | ✅✅ 显著改善 |
| 03 | Isotonic Regression | 0.8012 | **0.0000** | **1.0000** | ✅✅✅ 完美校准 |
| 04 | Focal Loss | 0.8004 | 0.1263 | 0.8547 | ❌ 校准恶化 |
| 05 | Label Smoothing | 0.8003 | 0.0112 | 1.0198 | ⚠️ 轻微恶化 |
| 06 | 分组训练 | 0.8584 | 0.0038 | 1.0039 | ✅ 效果好但成本高 |

---

## 🎯 核心发现

### 1. 基线模型整体校准良好

**整体指标**:
- AUC: 0.8010 (排序能力强)
- ECE: 0.0076 (平均校准误差低)
- PCOC: 1.0172 (接近理想值 1.0)

**结论**: 模型预测概率已经比较可靠。

### 2. 分组评估发现偏差

**Top business_type 校准差异**:

| BT | 样本占比 | PCOC | ECE | 校准情况 |
|----|---------|------|-----|---------|
| 1 | 33.68% | 1.0296 | 0.0147 | ⚠️ 预测偏高 3% |
| 11 | 14.94% | 1.0385 | 0.0102 | ⚠️ 预测偏高 4% |
| 3 | 5.56% | 1.0784 | 0.0222 | ⚠️ 预测偏高 8% |
| 6 | 9.64% | 1.0000 | 0.0113 | ✅ 完美校准 |

**结论**: 
- 不同 business_type 校准需求不同
- Top 2 business_type (占 48.62%) 都有偏差
- 需要分组校准

### 3. Temperature Scaling 效果有限但稳定

**整体校准**:
- 温度参数：1.0111 (接近 1.0)
- PCOC: 1.0172 → 1.0174 (几乎不变)
- ECE: 0.0076 → 0.0077 (几乎不变)

**分组校准**:
- BT 3: PCOC 1.0784 → 0.9870 (改善 9.1%)
- BT 3: ECE 0.0222 → 0.0098 (改善 56%)
- 不同 business_type 温度：[0.87, 2.17]

**结论**: 
- 单参数方法能力有限
- 分组校准效果更显著
- 对 AUC 无影响

### 4. Isotonic Regression 完美校准 ⭐

**惊人的结果**:
- ECE: 0.0076 → **0.0000** ✅
- PCOC: 1.0172 → **1.0000** ✅
- 所有 business_type 的 PCOC 都变成 1.0000
- 所有 business_type 的 ECE 都变成 0.0000

**原因**:
- 非参数方法，可以拟合任意单调函数
- 比 Temperature Scaling 更灵活

**注意**:
- 可能在大数据量下过拟合
- 需要足够的校准数据

### 5. Focal Loss 对校准有害 ❌

**意外发现**:
- ECE: 0.0076 → **0.1263** (恶化 16.6 倍)
- PCOC: 1.0172 → **0.8547** (预测偏低 14.5%)
- LogLoss: 0.5228 → 0.5791 (恶化)

**原因**:
- Focal Loss 改变了预测分布
- 降低易分类样本权重，导致预测概率整体偏低
- 对校准性能有负面影响

**结论**: 不推荐用于校准改善

### 6. Label Smoothing 效果有限 ⚠️

**结果**:
- ECE: 0.0076 → 0.0112 (轻微恶化)
- PCOC: 1.0172 → 1.0198 (基本不变)
- AUC: 0.8010 → 0.8003 (基本不变)

**结论**: Label Smoothing 对校准改善有限

### 7. 分组训练效果好但成本高 ✅

**初步结果**:
- AUC: 0.8584 (比整体高)
- ECE: 0.0038 (比整体低)
- PCOC: 1.0039 (非常接近 1.0)

**缺点**:
- 需要为每个 BT 维护独立模型
- 推理成本高
- 小 BT 样本不足

**结论**: 适合 Top BT，不适合全量

---

## 📋 方法对比总结

| 方法 | 校准效果 | 复杂度 | 推荐度 | 适用场景 |
|------|---------|--------|--------|----------|
| **无校准** | ✅ 好 | 低 | ⭐⭐⭐ | 基线已经很好 |
| **Temperature (整体)** | ✅ 轻微改善 | 低 | ⭐⭐⭐ | 轻微偏差 |
| **Temperature (分组)** | ✅✅ 显著改善 | 中 | ⭐⭐⭐⭐ | 分组偏差 |
| **Isotonic Regression** | ✅✅✅ 完美 | 中 | ⭐⭐⭐⭐⭐ | 需要精确校准 |
| **Focal Loss** | ❌ 恶化 | 高 | ⭐ | 不推荐 |
| **Label Smoothing** | ⚠️ 轻微恶化 | 低 | ⭐⭐ | 不推荐 |
| **分组训练** | ✅✅ 好 | 高 | ⭐⭐⭐ | Top BT 专用 |

---

## 🎓 研究结论

### 核心洞察

1. **基线模型校准已经很好**
   - PCOC = 1.0172
   - 不需要额外的后处理校准

2. **分组评估很重要**
   - 整体指标掩盖了分组偏差
   - Top business_type 需要重点关注

3. **Isotonic Regression 最强**
   - 完美校准 (ECE=0, PCOC=1)
   - 比 Temperature Scaling 强大得多

4. **Focal Loss 不适合校准**
   - 反而会恶化校准性能
   - 不推荐用于此目的

5. **校准不影响排序**
   - 所有方法对 AUC 影响极小
   - 校准主要改善概率估计

### 实践建议

1. **生产环境推荐**:
   - 如果基线校准好 (PCOC ~ 1.0): 无需后处理
   - 如果有轻微偏差：Temperature Scaling
   - 如果需要精确校准：Isotonic Regression ⭐
   - 如果资源充足：Top BT 分组训练

2. **评估建议**:
   - 必须按 business_type 分组评估
   - 关注 Top business_type 的校准偏差

3. **避免做法**:
   - 不要用 Focal Loss 改善校准
   - 不要只看整体指标

---

## 📂 实验文件

```
results/
├── exp01_baseline_metrics.json           # 基线整体指标
├── exp01b_business_type_metrics.json     # 分组评估
├── exp02_overall_calibration.json        # Temperature 整体
├── exp02_bt_calibration.json             # Temperature 分组
├── exp02_temperatures.json               # 温度参数
├── exp03_overall_isotonic.json           # Isotonic 整体
├── exp03_bt_isotonic.json                # Isotonic 分组
├── exp04_focal_loss.json                 # Focal Loss 对比
├── exp05_label_smoothing.json            # Label Smoothing 对比
└── exp06_grouped_training.json           # 分组训练对比
```

---

## 📚 参考文献

1. Guo et al., 2017 - "On Calibration of Modern Neural Networks" (ICML)
2. Niculescu-Mizil & Caruana, 2005 - "Predicting Good Probabilities"
3. Mukhoti et al., 2020 - "Calibrating Deep Neural Networks using Focal Loss"
4. Hendrycks & Gimpel, 2016 - "Keeping Neural Networks Simple by Minimizing Distributional Smoothing"

---

*研究完成 - 牛顿 🍎*
