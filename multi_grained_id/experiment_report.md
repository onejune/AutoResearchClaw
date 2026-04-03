# Multi-Grained ID Embedding 实验报告

## 项目概述

- **项目名称**: Multi-Grained ID Embedding for CTR/CVR Prediction
- **数据集**: IVR Sample v16 CTCVR (训练集 326.5 万样本，测试集 123.1 万样本)
- **任务**: CTCVR 预测（点击后转化）
- **Baseline AUC**: 0.8459, PCOC: 1.0821

## 实验结果汇总

### 第一批实验（exp001-008）

| 实验 | 方法 | AUC | PCOC | vs Baseline |
|------|------|-----|------|-------------|
| Baseline | 共享基线 | **0.8459** | 1.0821 | - |
| exp002 | DDS | **0.8459** | **0.9833** ✅ | 校准性最佳 |
| exp004 | MetaEmb | **0.8462** 🏆 | 1.1762 | AUC 最高 |
| exp005 | Contrastive | 0.8457 | 1.1250 | -0.2‱ |
| exp003 | Hierarchical | 0.8454 | 1.0014 | -0.5‱ |
| exp007 | Combined | 0.8419 | 1.0316 | -4.0‱ |
| exp001 | AutoEmb | - | - | 未完成 |
| exp006 | FiBiNET | 0.8244 | 1.0840 | **-21.5‱** ❌ |

### 第二批实验：Hierarchical 优化（exp009-014）

| 实验 | 方法 | AUC | PCOC | vs Baseline |
|------|------|-----|------|-------------|
| exp014 | DDS 分层维度 | 0.8432 | 1.0972 | -2.7‱ |
| exp011 | Attention | 0.8425 | 1.0555 | -3.4‱ |
| exp012 | 对比学习 | 0.8422 | 1.0577 | -3.7‱ |
| exp013 | 动态权重 | 0.8416 | 1.1350 | -4.3‱ |
| exp010 | 门控传递 | 0.8415 | 1.0549 | -4.4‱ |
| exp009 | MetaEmb 集成 | 0.5074 | 4.38 | ❌ 失败 |

## 关键发现

### 1. AUC 表现
- **MetaEmb 最高**（0.8462），比 Baseline 提升 0.3 千分点
- **Hierarchical 系列均未能超越 Baseline**
- **FiBiNET 效果最差**（比 Baseline 低 21.5 千分点）

### 2. 校准性（PCOC）表现
- **DDS 校准性最佳**（PCOC=0.9833），几乎完美校准
- Hierarchical 方法的 PCOC 更稳定（1.05-1.14）
- Baseline PCOC=1.0821，略有高估

### 3. Hierarchical 优化结论
- 门控传递、Attention、对比学习、动态权重、DDS 分层维度 均未带来 AUC 提升
- 可能原因：数据量不足以学习复杂的层级关系
- Hierarchical 方法更适合数据量更大、ID 层级更丰富的场景

## 技术细节

### 数据集
- 特征：125 个类别特征
- 标签：ctcvr_label（mean=0.1137）
- 关键层级结构：campaignid → campaignsetid → offerid → demand_pkgname → business_type

### 模型配置
- Embedding 维度：64
- DNN 结构：[256, 128]
- 优化器：Adam (lr=5e-5)
- 训练：1 epoch

### 实验规范
- 所有实验必须报告 AUC + PCOC
- 只跑 1 epoch（避免过拟合）
- 使用 HardwareMonitor 智能路由 GPU/CPU

## 结论

1. **MetaEmb** 在 AUC 上略有提升，推荐作为生产模型
2. **DDS** 在校准性上表现最佳，适合对校准要求高的场景
3. **Hierarchical** 方法需要更大数据量才能发挥效果
4. **FiBiNET** 在此数据集上不适用

---

*更新时间: 2026-04-02*