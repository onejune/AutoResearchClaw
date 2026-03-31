# Isotonic Regression 扩展方法研究笔记

## 📚 核心概念回顾

**Isotonic Regression (单调回归)**:
- 非参数方法，拟合单调递增函数
- 完美校准 (ECE=0, PCOC=1)
- 可能过拟合，需要足够数据

---

## 🔬 扩展方法

### 1. **Pool Adjacent Violators Algorithm (PAVA)** - 标准实现

**原理**: 
- 经典算法，O(n) 时间复杂度
- 通过合并违反单调性的相邻点来构建单调函数

**优点**:
- 计算高效
- 保证全局最优解

**缺点**:
- 对噪声敏感
- 容易过拟合

**适用场景**: 大数据量，需要精确校准

---

### 2. **Regularized Isotonic Regression** - 正则化版本

**原理**:
- 在单调约束下添加平滑正则项
- 常见正则化：L2、总变差 (Total Variation)

**公式**:
```
min Σ(y_i - f(x_i))² + λ·R(f)
s.t. f 是单调递增的
```

**优点**:
- 减少过拟合
- 更平滑的校准曲线

**缺点**:
- 需要调参 (λ)
- 计算复杂度增加

**推荐**: ⭐⭐⭐⭐⭐ 生产环境首选

---

### 3. **Concave/Convex Isotonic Regression** - 凹/凸约束

**原理**:
- 额外添加凹性或凸性约束
- Concave: 边际收益递减 (适合概率校准)
- Convex: 边际收益递增

**优点**:
- 更符合概率分布特性
- 进一步减少过拟合

**缺点**:
- 约束更强，可能欠拟合

**适用场景**: 
- Concave: CTR/CVR 预测校准
- Convex: 风险评分校准

---

### 4. **Quantile Isotonic Regression** - 分位数校准

**原理**:
- 对不同分位区间分别做单调回归
- 例如：[0, 0.1], [0.1, 0.3], [0.3, 0.7], [0.7, 1.0]

**优点**:
- 可以针对不同置信度区间优化
- 高置信度区域更精确

**缺点**:
- 边界处可能不连续
- 需要更多数据

**适用场景**: 高风险决策 (关注高置信度样本)

---

### 5. **Grouped Isotonic Regression** - 分组单调回归

**原理**:
- 按业务维度分组 (如 business_type)
- 每组独立训练 Isotonic Regression

**优点**:
- 考虑业务差异
- 各组都能达到完美校准

**缺点**:
- 小样本组效果差
- 维护成本高

**适用场景**: Top business_type 专用模型

---

### 6. **Bayesian Isotonic Regression** - 贝叶斯版本

**原理**:
- 将单调函数建模为随机过程
- 输出校准分布而非点估计

**优点**:
- 提供不确定性量化
- 对小样本更鲁棒

**缺点**:
- 计算复杂度高
- 推理慢

**适用场景**: 需要置信区间的场景

---

### 7. **Deep Calibration Layers** - 深度校准层

**原理**:
- 用神经网络替代 Isotonic Regression
- 添加单调性约束 (Monotonic Neural Networks)

**实现方式**:
```python
class MonotonicCalibrationLayer(nn.Module):
    def forward(self, x):
        # 确保导数非负
        return F.softplus(weights) * x + bias
```

**优点**:
- 可端到端训练
- 与其他模块集成方便

**缺点**:
- 训练不稳定
- 需要大量数据

**推荐**: ⭐⭐⭐ 实验性质

---

### 8. **Two-Stage Calibration** - 两阶段校准

**原理**:
- Stage 1: Temperature Scaling (粗略校准)
- Stage 2: Isotonic Regression (精细校准)

**优点**:
- 结合两者优势
- 更稳定

**缺点**:
- 两次后处理
- 延迟增加

**适用场景**: 离线批量校准

---

### 9. **Online Isotonic Regression** - 在线更新

**原理**:
- 支持增量更新
- 适应数据分布漂移

**算法**:
- PAVA 的在线版本
- 滑动窗口更新

**优点**:
- 实时适应
- 无需重新训练

**缺点**:
- 理论保证弱
- 实现复杂

**适用场景**: 流式数据，概念漂移

---

### 10. **Multi-task Isotonic Calibration** - 多任务联合校准

**原理**:
- 同时校准多个相关任务 (CTR + CVR)
- 共享单调性约束

**优点**:
- 利用任务相关性
- 小任务受益于大任务

**缺点**:
- 任务冲突时效果差

**适用场景**: 多任务学习框架

---

## 📊 方法对比总结

| 方法 | 校准效果 | 复杂度 | 抗过拟合 | 推荐度 | 适用场景 |
|------|---------|--------|---------|--------|----------|
| **标准 Isotonic** | ✅✅✅ | 低 | ⚠️ | ⭐⭐⭐⭐ | 大数据量 |
| **Regularized** | ✅✅✅ | 中 | ✅✅✅ | ⭐⭐⭐⭐⭐ | 生产环境 |
| **Concave** | ✅✅✅ | 中 | ✅✅ | ⭐⭐⭐⭐ | CTR/CVR |
| **Quantile** | ✅✅✅ | 中 | ✅ | ⭐⭐⭐ | 高风险决策 |
| **Grouped** | ✅✅✅ | 高 | ✅✅ | ⭐⭐⭐⭐ | Top BT |
| **Bayesian** | ✅✅✅ | 高 | ✅✅✅ | ⭐⭐⭐ | 需要不确定性 |
| **Deep Layer** | ✅✅ | 高 | ⚠️ | ⭐⭐ | 实验 |
| **Two-Stage** | ✅✅✅ | 中 | ✅✅ | ⭐⭐⭐⭐ | 离线校准 |
| **Online** | ✅✅ | 中 | ✅ | ⭐⭐⭐ | 流式数据 |
| **Multi-task** | ✅✅ | 高 | ✅ | ⭐⭐⭐ | 多任务学习 |

---

## 🎯 实践建议

### 生产环境推荐方案

**方案 A (简单)**: Regularized Isotonic Regression
- 添加 L2 正则化
- 交叉验证选择 λ
- 每月重新训练一次

**方案 B (进阶)**: Grouped + Regularized
- Top 3 BT: 独立 Regularized Isotonic
- 其他 BT: 整体 Regularized Isotonic

**方案 C (高级)**: Two-Stage + Online
- Stage 1: Temperature Scaling (每日更新)
- Stage 2: Isotonic (每周更新)

---

## 📚 参考文献

1. **Brunk et al., 1968** - "Minimum concave interpolation" (PAVA 算法)
2. **Hildreth, 1954** - "Point estimates of isotonic regression"
3. **Turner & Safavi-Naini, 2009** - "Regularized isotonic regression"
4. **Zhang et al., 2021** - "Deep monotonic calibration networks"
5. **Kuleshov et al., 2018** - "Accurate uncertainties for deep learning using calibrated regression"

---

*研究笔记 - 牛顿 🍎*
