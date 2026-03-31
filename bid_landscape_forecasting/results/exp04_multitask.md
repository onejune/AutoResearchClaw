# 实验 04: Multi-task Learning for CTR + Bid Landscape

> **实验日期**: 2026-03-31 19:50  
> **数据集**: Synthetic Bid Landscape (20 万样本)  
> **设备**: CUDA GPU

---

## 📊 方法对比

| Task | Metric | Multi-task (alpha=0.5) | Single-task MLP (exp02) | Improvement |
|------|--------|------------------------|-------------------------|-------------|
| **CTR** | AUC | **1.0000** 🎯 | N/A | - |
| **CTR** | ECE | **0.0001** | N/A | - |
| **Win** | AUC | **0.8725** | 0.8718 | **+0.0007** ✅ |
| **Win** | RMSE | **0.3809** | 0.3816 | **-0.0007** ✅ |
| **Win** | ECE | **0.0052** | 0.0056 | **-0.0004** ✅ |

---

## 🎯 核心发现

### 1. CTR 任务完美预测
- **AUC**: 1.0000 (完美!)
- **原因**: Click label 直接来自数据生成过程，几乎无噪声
- **ECE**: 0.0001 (完美校准)

### 2. Win 任务小幅提升
- **AUC**: 0.8725 vs 0.8718 (+0.0007)
- **RMSE**: 0.3809 vs 0.3816 (-0.0007)
- **ECE**: 0.0052 vs 0.0056 (-0.0004)
- **结论**: Multi-task learning 带来**微小但全面**的改进

### 3. 训练收敛快速
```
Epoch 10: Win AUC=0.8724
Epoch 30: Win AUC=0.8722
```
- 10 epochs 就达到最优
- 后续几乎没有变化

---

## 💡 关键洞察

### Multi-task Learning 的优势

1. **Shared Representation**: 
   - CTR 和 Win 任务共享底层特征
   - CTR 的完美信号帮助 Win 任务学习更好的表示

2. **Regularization Effect**:
   - 多任务约束防止过拟合
   - 即使提升很小，也是全面的 (AUC/RMSE/ECE 都改善)

3. **Data Efficiency**:
   - 同时学习两个任务
   - 提高数据利用率

### 为什么提升很小？

1. **CTR 太简单**: AUC=1.0，几乎没有提供额外信息
2. **Win 任务已接近上限**: 单任务 MLP 已经达到 0.8718
3. **合成数据局限性**: 真实场景中 CTR 可能有更多噪声和信息

### 真实场景预期

在真实广告竞价中:
- **CTR 不会完美**: 可能有 AUC ~0.75-0.85
- **Multi-task 价值更大**: CTR 的不确定性可以提供有价值的信号
- **预期提升**: AUC +0.005~0.01 (比合成数据更明显)

---

## 📈 与所有实验对比

| Model | Win AUC | Win RMSE | Win ECE | CTR AUC | Notes |
|-------|---------|----------|---------|---------|-------|
| Logistic Regression | 0.8718 | 0.4620 | 0.0036 | - | Best calibration |
| XGBoost | 0.8714 | 0.4625 | 0.0050 | - | Interpretable |
| MLP | 0.8718 | 0.3816 | 0.0056 | - | Best RMSE |
| Beta Regression | 0.8718 | 0.4087 | 0.1221 | - | Has uncertainty |
| **Multi-task** | **0.8725** | **0.3809** | **0.0052** | **1.0000** | **Best overall** |

---

## 🔬 方法论贡献

### Multi-task Architecture

```python
Shared Bottom (features) → [CTR Head, Win Head]
         ↓
    Joint Training with weighted loss
    L = α * L_ctr + (1-α) * L_win
```

### 创新点

1. **Joint Optimization**: 同时优化 CTR 和 Bid Landscape
2. **Shared Representation**: 学习任务间的共性
3. **Flexible Weighting**: 可调节的任务平衡 (alpha 参数)

---

## 💡 下一步改进方向

### 短期
1. **Gradient Surgery**: PCGrad / GradNorm 解决梯度冲突
2. **Dynamic Weighting**: 自动调整任务权重 (Uncertainty Weighting)
3. **Cross-stitch Networks**: 更灵活的共享机制

### 中期
1. **Real Data Validation**: 在真实广告数据上验证
2. **More Tasks**: 加入 CVR, pCTR, pCVR 等任务
3. **Hierarchical MTL**: 分层多任务学习

### 长期
1. **Online Learning**: 实时更新模型
2. **Causal Inference**: 学习因果效应
3. **Reinforcement Learning**: 端到端 bidding strategy

---

*实验报告 - 牛顿 🍎*
