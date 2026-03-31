# Bid Landscape Forecasting Research

## 研究概述

**课题**: 广告竞价中的 Bid Landscape 预测  
**目标会议**: KDD / WWW / RecSys 2026  
**数据集**: IVR Sample v16 (待确认)  

---

## 研究背景

Bid Landscape 是广告竞价系统中的核心概念，描述了不同出价下的获胜概率分布。准确预测 Bid Landscape 可以:

1. **优化出价策略**: 在预算约束下最大化曝光/转化
2. **提升 ROI**: 避免过高或过低出价
3. **实时决策**: 支持毫秒级 bidding decision

---

## 研究方向

### Phase 1: Baseline & Data Exploration
- [ ] 数据质量检查 (样本量、特征分布、标签统计)
- [ ] 探索性数据分析 (EDA)
- [ ] 基线模型 (Linear Regression, XGBoost)

### Phase 2: Deep Learning Models
- [ ] MLP with different architectures
- [ ] TabNet / FT-Transformer
- [ ] Graph Neural Networks (if applicable)

### Phase 3: Advanced Methods
- [ ] Distribution Modeling (Quantile Regression)
- [ ] Multi-task Learning (CTR + CVR + Landscape)
- [ ] Online Learning / Adaptive Updates

---

## 项目结构

```
bid_landscape_forecasting/
├── README.md                 # 项目说明
├── config.researchclaw.yaml  # AutoResearchClaw 配置
├── data/                     # 数据处理模块
├── models/                   # 模型定义
├── experiments/              # 实验脚本
├── results/                  # 实验结果
└── references/               # 参考文献与笔记
```

---

## 快速开始

```bash
# 使用 AutoResearchClaw 自动研究
cd /mnt/workspace/open_research/autoresearch/bid_landscape_forecasting
researchclaw run --config config.researchclaw.yaml
```

---

*研究启动 - 2026-03-31*
