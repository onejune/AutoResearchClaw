# 实验笔记 - CTR/CVR 预估校准研究

## 2026-03-31 项目初始化

### 数据集分析

已完成数据质量检查：

```
数据集: IVR Sample v16 CTCVR
训练集: 3,265,331 样本
特征数: 126 (全部类别特征)

标签分布:
- CTR (click_label): 41.49%
- CTCVR (ctcvr_label): 11.37%
- 估算 CVR: 27.41%

词表大小:
- 最小: 3
- 最大: 2,383,974 (deviceid)
- 平均: 19,883
- 中位数: 17

数据质量: ✅ 通过
```

### AutoResearchClaw 配置

已创建 `config.researchclaw.yaml`:

- 模式: full-auto
- 主模型: qwen3-coder-plus
- 实验模式: sandbox
- 数据集: IVR Sample v16 CTCVR
- 特征处理: 无（全部当作类别特征）

### 核心指标

| 指标 | 说明 |
|------|------|
| ECE | Expected Calibration Error |
| MCE | Maximum Calibration Error |
| PCOC | Predicted/Observed CTR |
| AUC | 排序能力 |

### 实验计划

1. **Exp01**: 基线模型（无校准）
2. **Exp02**: Temperature Scaling
3. **Exp03**: Isotonic Regression
4. **Exp04**: Focal Loss
5. **Exp05**: 多任务联合校准

### 待办事项

- [x] 创建项目目录
- [x] 配置 AutoResearchClaw
- [x] 数据质量检查
- [ ] 初始化 AutoResearchClaw
- [ ] 运行研究流程
- [ ] 分析结果

---

## 研究假设

### H1: 深度学习模型过度自信

假设：模型预测概率偏高或偏低，与真实概率不一致

验证方法：
- 计算基线模型的 ECE, PCOC
- 绘制 Reliability Diagram

### H2: Temperature Scaling 最有效

假设：单参数调整是最简单有效的校准方法

验证方法：
- 对比 Temperature, Isotonic, Histogram 的 ECE 下降
- 分析校准后的 AUC 变化

### H3: Focal Loss 改善训练时校准

假设：Focal Loss 可以在训练时改善校准

验证方法：
- 对比 Focal Loss vs BCE 的 ECE
- 分析训练曲线

### H4: 多任务联合校准优于独立校准

假设：CTR + CVR 联合校准效果更好

验证方法：
- 对比独立校准 vs 联合校准
- 分析任务间的关系

---

*笔记维护：牛顿 🍎*
