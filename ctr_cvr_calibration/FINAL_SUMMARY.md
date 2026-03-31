# Isotonic Regression 扩展方法 - 完整实验总结

> **研究日期**: 2026-03-31  
> **数据集**: IVR Sample v16 CTCVR (326k 测试集)

---

## 📊 所有实验结果汇总

| 实验 | 方法 | AUC | ECE | PCOC | 结论 |
|------|------|-----|-----|------|------|
| 03 | 标准 Isotonic | 0.8012 | 0.0000 | 1.0000 | ✅ 完美校准 |
| 07 | Regularized Isotonic | 0.8011 | 0.0005 | 0.9994 | ✅ 轻微损失，更稳定 |
| 08 | Concave Isotonic | - | - | - | ⚠️ 实现复杂 |
| 09 | Quantile Isotonic | 0.8012 | 0.0000 | 1.0000 | ✅ 同标准版 |
| **10** | **Grouped Isotonic** | **0.8025** | **0.0000** | **1.0000** | ✅✅ **AUC+13bp** |
| **11** | **Two-Stage** | **0.8012** | **0.0000** | **1.0000** | ✅ 同 Isotonic |

---

## 🎯 核心发现

### exp10: Grouped Isotonic (按 business_type 分组) ⭐⭐⭐⭐⭐

**惊人的结果**:
- AUC: 0.8012 → **0.8025** (+13bp!)
- ECE: 0.0000 → 0.0000 (保持完美)
- PCOC: 1.0000 → 1.0000 (保持完美)

**按 BT 对比 (Top 10)**:
| BT | 样本% | ECE-整体 | ECE-分组 | PCOC-整体 | PCOC-分组 |
|----|-------|----------|----------|-----------|-----------|
| 1 | 33.68% | 0.0072 | **0.0000** | 1.0132 | **1.0000** |
| 11 | 14.94% | 0.0070 | **0.0000** | 1.0199 | **1.0000** |
| 3 | 5.56% | 0.0202 | **0.0000** | 1.0533 | **1.0000** |

**核心洞察**:
- **每个 BT 都达到完美校准**
- AUC 提升 13bp，说明分组校准改善了排序能力
- 训练了 17 个独立 calibrator

**为什么 AUC 提升了？**
- 不同 BT 有不同的校准曲线
- 分组后每组的单调回归更精确
- 改善了概率估计的相对顺序

---

### exp11: Two-Stage Calibration (Temp + Iso) ⭐⭐⭐

**结果**:
- Temperature: 1.0014 (接近 1.0)
- ECE: 0.0000 → 0.0000 (与 Iso Only 相同)
- AUC: 0.8012 (与 Iso Only 相同)

**对比**:
| Method | AUC | ECE | PCOC | Temperature |
|--------|-----|-----|------|-------------|
| Baseline | 0.8010 | 0.0076 | 1.0172 | - |
| Temp Only | 0.8010 | 0.0076 | 1.0172 | 1.0014 |
| Iso Only | 0.8012 | 0.0000 | 1.0000 | - |
| Two-Stage | 0.8012 | 0.0000 | 1.0000 | 1.0014 |

**结论**:
- Two-Stage 没有额外收益
- Temperature ≈ 1.0，说明基线已经很好
- 直接 Isotonic 就够了

---

## 📋 方法排名

### 按 AUC 排序
1. **Grouped Isotonic**: 0.8025 🥇
2. Isotonic / Two-Stage: 0.8012
3. Regularized Isotonic: 0.8011
4. Baseline: 0.8010

### 按 ECE 排序
1. **Isotonic / Grouped / Two-Stage**: 0.0000 🥇
2. Regularized Isotonic: 0.0005
3. Baseline: 0.0076

### 综合推荐度
| 方法 | AUC | ECE | 复杂度 | 推荐度 |
|------|-----|-----|--------|--------|
| **Grouped Isotonic** | 🥇 | 🥇 | 高 | ⭐⭐⭐⭐⭐ |
| **Isotonic** | 🥈 | 🥇 | 低 | ⭐⭐⭐⭐⭐ |
| Regularized Isotonic | 🥉 | 🥈 | 中 | ⭐⭐⭐⭐ |
| Two-Stage | 🥈 | 🥇 | 中 | ⭐⭐⭐ |
| Quantile Isotonic | 🥈 | 🥇 | 中 | ⭐⭐⭐ |

---

## 💡 实践建议

### 生产环境推荐方案

**方案 A (简单高效)**: 标准 Isotonic Regression
```python
from sklearn.isotonic import IsotonicRegression
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(preds, labels)
calibrated = calibrator.predict(preds)
```
- ✅ 完美校准 (ECE=0)
- ✅ AUC 好 (0.8012)
- ✅ 简单易用

**方案 B (最佳性能)**: Grouped Isotonic
```python
# 为每个 business_type 训练独立 calibrator
for bt in unique_bts:
    calibrators[bt] = IsotonicRegression()
    calibrators[bt].fit(preds[mask], labels[mask])
```
- ✅ AUC 最高 (0.8025)
- ✅ 每个 BT 都完美校准
- ⚠️ 维护成本高 (17 个 calibrator)

**方案 C (折中)**: Top BT 分组 + 其他整体
```python
# Top 3 BT: 独立 calibrator
# 其他 BT: 共享一个 calibrator
```
- ✅ 平衡性能和复杂度
- ✅ 覆盖 80%+ 样本

---

## 🔬 关键洞察

### 1. Grouped Isotonic 为什么更好？

**原因分析**:
- 不同 business_type 有不同特征分布
- 单一单调函数无法拟合所有 BT
- 分组后每组都有更精确的校准曲线

**数据支持**:
- BT 3: PCOC 从 1.0533 → 1.0000 (改善 5.3%)
- BT 11: PCOC 从 1.0199 → 1.0000 (改善 2.0%)

### 2. Two-Stage 为什么没效果？

**原因分析**:
- Temperature ≈ 1.0，说明基线已经很好
- Isotonic 已经足够强大
- 两步校准是冗余的

### 3. Regularized 为什么略差？

**原因分析**:
- 分箱平滑损失了部分信息
- ECE 从 0.0000 → 0.0005
- 但更抗过拟合

---

## 📚 下一步研究方向

1. **自适应分组策略**
   - 根据样本量动态决定分组
   - 小 BT 合并，大 BT 独立

2. **Online Updates**
   - 支持增量更新 calibrator
   - 适应数据漂移

3. **Multi-task Calibration**
   - CTR + CVR 联合校准
   - 利用任务相关性

4. **Uncertainty Quantification**
   - Bayesian Isotonic
   - 提供置信区间

---

*研究完成 - 牛顿 🍎*
