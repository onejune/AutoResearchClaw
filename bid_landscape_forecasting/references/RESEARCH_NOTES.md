# Bid Landscape Forecasting - 研究笔记

## 问题定义

**Bid Landscape**: 给定广告请求的特征，预测不同出价下的获胜概率分布

数学形式化:
- 输入：用户特征 x, 广告特征 a, 上下文特征 c
- 输出：P(win | bid) = F(bid; x, a, c)
- 目标：学习条件 CDF (Cumulative Distribution Function)

## 数据挑战

### 当前 IVR 数据集
- ✅ 326 万样本，CTR/CVR 标签
- ❌ **无 bid amount 特征**
- ❌ **无 win/loss 标签**

### 解决方案

#### 方案 A: 模拟 Bid Landscape (推荐用于启动)
从 CTR 数据推导:
1. 假设 CTR ≈ P(win) × P(click|win)
2. 如果 P(click|win) 相对稳定，则 CTR ∝ P(win)
3. 生成 synthetic bid data:
   - 对每个样本，假设一个"真实价值" v ~ Beta(α, β)
   - 生成多个 bid levels: b ∈ {0.5v, 0.7v, v, 1.3v, 1.5v}
   - 计算 win probability: P(win|b) = sigmoid(k × (b - v))

#### 方案 B: 寻找其他数据集
- Criteo Display Advertising Dataset (可能有 bid 信息)
- 内部广告平台日志 (如果有访问权限)
- 公开竞价数据集 (如 AdExchange logs)

#### 方案 C: Multi-task Learning
同时预测:
1. CTR (已有标签)
2. Bid Landscape (合成标签)
共享底层表示，互相增强

## 研究方向

### Phase 1: Data Simulation & Baseline (Week 1-2)
- [x] 数据质量检查
- [ ] 设计 bid landscape 生成策略
- [ ] 基线模型：
  - Linear Regression (bid → win prob)
  - Quantile Regression
  - XGBoost with custom objective

### Phase 2: Deep Learning Models (Week 3-4)
- [ ] MLP with different architectures
- [ ] TabNet (interpretable tabular DL)
- [ ] FT-Transformer (feature transformation)
- [ ] Distribution modeling:
  - Beta Regression (output α, β parameters)
  - Quantile Neural Network
  - Mixture Density Networks

### Phase 3: Advanced Methods (Week 5-6)
- [ ] Multi-task: CTR + Bid Landscape
- [ ] Online learning (FTRL, Online Newton Step)
- [ ] Calibration methods (Isotonic, Temperature Scaling)
- [ ] Uncertainty quantification (MC Dropout, Ensemble)

### Phase 4: Evaluation & Analysis (Week 7-8)
- [ ] Per-segment analysis (by business_type, advertiser)
- [ ] Ablation studies
- [ ] Real-world simulation (budget-constrained bidding)
- [ ] Comparison with industry baselines

## 评估指标

### Ranking Metrics
- **AUC**: 排序能力 (主要指标)
- **NDCG@K**: Top-K 排序质量

### Regression Metrics
- **RMSE**: 预测误差
- **MAE**: 平均绝对误差
- **R²**: 解释方差

### Calibration Metrics
- **ECE**: Expected Calibration Error
- **PCOC**: Probability Coverage
- **Brier Score**: 概率预测质量

### Business Metrics (Simulation)
- **Win Rate @ Budget**: 预算约束下的获胜率
- **ROI**: Return on Investment
- **Cost per Acquisition**: CPA

## 参考文献

### Core Papers
1. **Bid Landscape Modeling**
   - "Learning Bid Landscapes for Display Advertising" (KDD 2019)
   - "Deep Bid Landscape Prediction" (WWW 2021)
   
2. **Distribution Modeling**
   - "Quantile Regression Forests" (Meinshausen, 2006)
   - "Beta Regression for Proportional Data" (Ferrari & Cribari-Neto, 2004)
   
3. **Tabular Deep Learning**
   - "TabNet: Attentive Interpretable Tabular Learning" (KDD 2019)
   - "FT-Transformer: Feature Transformation Transformer" (Arxiv 2021)

### Related Work
- CTR/CVR prediction (DeepFM, xDeepFM, AutoInt)
- Calibration methods (Guo et al. 2017)
- Multi-task learning (MMOE, PLE)

## 关键挑战

1. **数据稀缺**: 真实 bid data 难以获取
   - 解决：高质量模拟 + multi-task learning
   
2. **分布偏移**: Train/test distribution mismatch
   - 解决：Domain adaptation, online updates
   
3. **实时性要求**: 毫秒级预测
   - 解决：Model distillation, feature selection
   
4. **可解释性**: 业务方需要理解模型决策
   - 解决：SHAP values, attention visualization

## 下一步行动

1. ✅ 项目目录创建完成
2. ✅ Config 配置完成
3. ⏳ 设计并实现 bid landscape 生成脚本
4. ⏳ 运行 AutoResearchClaw 自动研究
5. ⏳ 人工审核关键节点 (Stage 5, 9, 15, 20)

---

*Created: 2026-03-31*  
*Author: 牛顿 🍎*
