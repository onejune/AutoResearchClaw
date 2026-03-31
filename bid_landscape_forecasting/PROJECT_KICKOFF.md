# Bid Landscape Forecasting - 项目启动报告

> **启动日期**: 2026-03-31  
> **研究负责人**: 牛顿 🍎  
> **目标会议**: KDD / WWW / RecSys 2026

---

## ✅ 已完成工作

### 1. 项目目录创建
```
/mnt/workspace/open_research/autoresearch/bid_landscape_forecasting/
├── README.md                    # 项目说明
├── config.researchclaw.yaml     # AutoResearchClaw 配置
├── data/                        # 数据模块
│   └── bid_landscape_train_small.parquet  # 50 万样本 (测试用)
├── scripts/
│   └── generate_bid_data.py     # 数据生成脚本
├── references/
│   └── RESEARCH_NOTES.md        # 研究笔记
├── models/                      # 模型定义
├── experiments/                 # 实验脚本
└── results/                     # 实验结果
```

### 2. 数据准备

**挑战**: IVR 数据集无 bid amount 特征

**解决方案**: 从 CTR 数据合成 bid landscape
- 假设每个样本有"真实价值" v ~ Beta(α, β)
- 生成 5 个 bid levels: {0.5v, 0.7v, v, 1.3v, 1.5v}
- 计算 win probability: P(win|b) = sigmoid(k × (b - v))

**生成的数据**:
- 样本数：500,000 (10 万原始样本 × 5 bids)
- Win rate: 0.5002
- Bid range: [0.01, 2.68]
- 包含特征：bid_amount, win_label, win_prob, true_value, business_type

### 3. AutoResearchClaw 配置

**关键设置**:
```yaml
research:
  topic: "Bid Landscape Forecasting in Online Advertising"
  domains: ["machine-learning", "advertising", "recommendation-systems"]
  
experiment:
  mode: "sandbox"  # 本地执行真实代码
  metric_key: "auc"
  metric_direction: "maximize"
  gpu_required: true
  
llm:
  primary_model: "qwen3-coder-plus"  # 稳定可靠
```

**人工审核节点**: Stage 5, 9, 15, 20

---

## 📋 研究计划

### Phase 1: Baseline & Data Exploration (Week 1-2)
- [x] 数据质量检查
- [x] 合成 bid landscape 数据
- [ ] 探索性数据分析 (EDA)
- [ ] 基线模型：
  - Logistic Regression (bid → win prob)
  - Quantile Regression
  - XGBoost with custom objective

### Phase 2: Deep Learning Models (Week 3-4)
- [ ] MLP with different architectures
- [ ] TabNet (interpretable tabular DL)
- [ ] FT-Transformer
- [ ] Distribution modeling:
  - Beta Regression
  - Quantile Neural Network
  - Mixture Density Networks

### Phase 3: Advanced Methods (Week 5-6)
- [ ] Multi-task: CTR + Bid Landscape
- [ ] Online learning (FTRL)
- [ ] Calibration methods
- [ ] Uncertainty quantification

### Phase 4: Evaluation & Analysis (Week 7-8)
- [ ] Per-segment analysis
- [ ] Ablation studies
- [ ] Real-world simulation
- [ ] Paper writing

---

## 🎯 评估指标

| 类型 | 指标 | 说明 |
|------|------|------|
| **Ranking** | AUC | 排序能力 (主要指标) |
| **Regression** | RMSE, MAE, R² | 预测误差 |
| **Calibration** | ECE, PCOC, Brier Score | 概率校准 |
| **Business** | Win Rate @ Budget, ROI | 业务价值 |

---

## 🔬 关键技术点

### 1. 问题定义
**Bid Landscape**: P(win | bid, features) 的条件 CDF

### 2. 数据策略
- **短期**: 高质量合成数据 (已完成)
- **中期**: Multi-task learning (CTR + Bid)
- **长期**: 获取真实 bid logs

### 3. 模型选择
- **Baseline**: Linear, XGBoost
- **Deep Learning**: TabNet, FT-Transformer
- **Distribution**: Beta Regression, Quantile NN

### 4. 评估重点
- **整体性能**: AUC, RMSE
- **分群分析**: by business_type, advertiser
- **业务模拟**: budget-constrained bidding

---

## 🚀 下一步行动

1. **立即**: 运行 AutoResearchClaw 自动研究
   ```bash
   cd /mnt/workspace/open_research/autoresearch/bid_landscape_forecasting
   researchclaw run --config config.researchclaw.yaml
   ```

2. **监控**: 关注 Stage 5 (文献调研完成), Stage 9 (实验设计), Stage 15 (初步结果), Stage 20 (论文草稿)

3. **干预**: 
   - 如果研究方向偏离，使用 `researchclaw steer`
   - 如果数据有问题，补充更多合成数据
   - 如果模型效果差，调整 architecture

---

## 📚 参考文献

### Core Papers
1. "Learning Bid Landscapes for Display Advertising" (KDD 2019)
2. "Deep Bid Landscape Prediction" (WWW 2021)
3. "TabNet: Attentive Interpretable Tabular Learning" (KDD 2019)

### Related Work
- CTR/CVR prediction (DeepFM, xDeepFM)
- Calibration methods (Guo et al. 2017)
- Distribution modeling (Quantile Regression, Beta Regression)

---

## ⚠️ 风险与挑战

1. **数据真实性**: 合成数据可能与真实分布有偏差
   - 缓解：multi-task learning, domain adaptation

2. **实时性要求**: 毫秒级预测
   - 缓解：model distillation, feature selection

3. **可解释性**: 业务方需要理解模型
   - 缓解：SHAP values, attention visualization

---

*项目启动 - 2026-03-31*  
*牛顿 🍎*
