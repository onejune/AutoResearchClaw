# LTV Optimization Research - Implementation Plan

## 研究启动说明

由于 AutoResearchClaw 需要额外的 LLM API 配置，我们将采用混合研究方法:

1. **文献调研**: 手动深入研究 4 篇核心论文
2. **代码实现**: 基于论文逐步实现各方法
3. **实验对比**: 在 Taobao 数据集上进行统一评估
4. **分析总结**: 形成系统性研究报告

## 第一阶段：文献深度理解 (Week 1)

### 1.1 ZILN - Zero-Inflated Lognormal (Google, 2019)

**论文**: https://arxiv.org/pdf/1912.07753

**核心公式**:
```
P(y) = π * δ₀(y) + (1-π) * LogNormal(y|μ, σ)

其中:
- π: 用户不付费的概率 (zero-inflation component)
- δ₀: Dirac delta function at zero
- LogNormal: 对数正态分布参数 (μ, σ)

损失函数 (Negative Log-Likelihood):
L = -Σ [I(yᵢ=0) * log(πᵢ) + I(yᵢ>0) * log((1-πᵢ) * f_LogNormal(yᵢ|μᵢ, σᵢ))]
```

**模型架构**:
```
Input Features → DNN Shared Layers
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
      Sigmoid     Linear      Linear (softplus)
        ↓           ↓           ↓
       π_hat      μ_hat       σ_hat
        ↓           ↓           ↓
    Zero-Inflated LogNormal Distribution
        ↓
    Negative Log-Likelihood Loss
```

**关键实现点**:
- 三输出头：π (sigmoid), μ (linear), σ (softplus 保证正数)
- 自定义 NLL 损失函数
- 需要处理数值稳定性 (log(0) 问题)

**待回答问题**:
- [ ] 如何初始化三个输出头？
- [ ] 梯度消失/爆炸问题如何处理？
- [ ] 相比两阶段方法的优势量化？

### 1.2 ODMN & MDME - Kuaishou, 2022

**论文**: https://arxiv.org/pdf/2208.13358

#### ODMN (Ordered Deep Multi-timeframe Network)

**核心思想**: 同时预测多个时间窗口的 LTV，利用有序依赖关系

**时间框架**: 通常预测 7 日、14 日、30 日 LTV

**有序约束**:
```
LTV_7d ≤ LTV_14d ≤ LTV_30d  (累积 LTV 天然有序)

损失函数加入单调性约束:
L_ordered = λ * Σ ReLU(LTV_t₁ - LTV_t₂ + ε)  for t₁ < t₂
```

**架构**:
```
Input → Shared Embedding → Timeframe-specific Towers
                                    ↓
                         [LTV_7d, LTV_14d, LTV_30d]
                                    ↓
                    Multi-task Loss + Ordering Constraint
```

#### MDME (Multi-Distribution Mixture Estimation)

**核心思想**: 分桶采样 + 混合分布专家

**分桶策略**:
```
Bucket 0: LTV = 0 (非付费用户)
Bucket 1: 0 < LTV ≤ threshold₁ (小 R)
Bucket 2: threshold₁ < LTV ≤ threshold₂ (中 R)
...
Bucket K: LTV > threshold_{K-1} (大 R)
```

**训练策略**:
- 每个 bucket 独立采样平衡
- 训练 K+1 个专家模型
- 推理时加权集成

**关键公式**:
```
P(LTV|x) = Σ w_k(x) * P_k(LTV|x)

其中 w_k(x) 是样本属于 bucket k 的门控概率
```

**待回答问题**:
- [ ] 最优 bucket 数量是多少？
- [ ] threshold 如何设定 (等分位数 vs 业务规则)?
- [ ] ODMN 和 MDME 能否结合使用？

### 1.3 ExpLTV - Tencent, 2023

**论文**: https://arxiv.org/pdf/2308.12729

**核心创新**: 大 R 用户检测作为 MoE 门控

**两阶段架构**:
```
Stage 1: Whale Detection (二分类)
    Input → Feature Tower → Whale/Non-Whale Classifier
    
Stage 2: Expert-based LTV Prediction
    Input → Feature Tower → Gating Network
                            ↓
              ┌─────────────┼─────────────┐
              ↓             ↓             ↓
          Whale Expert  Mixed Expert  Small User Expert
              ↓             ↓             ↓
            LTV_pred (Mixture of weighted experts)
```

**门控网络**:
```
g(x) = Softhead(W_g * h(x) + b_g)  # [num_experts]

LTV_pred = Σ g_i(x) * expert_i(x)
```

**大 R 定义**:
- 通常取 LTV 分布 top 1%-5% 用户
- 或基于业务规则 (如累计付费>阈值)

**待回答问题**:
- [ ] 门控网络与 whale detector 是否共享参数？
- [ ] 专家数量与性能的关系？
- [ ] 在线场景下如何更新 whale 定义？

### 1.4 CMLTV - Huawei, 2023

**论文**: https://arxiv.org/pdf/2306.14400

**核心思想**: 对比学习 + 多视角异构回归器集成

**对比学习框架**:
```
View 1: Original Features → Encoder → Representation z₁
View 2: Augmented Features → Encoder → Representation z₂

Contrastive Loss:
L_contrast = -log(exp(sim(z₁, z₂)/τ) / Σ exp(z₁, z_neg)/τ)
```

**数据增强策略**:
- Feature dropout
- Sample augmentation (SMOTE for minority class)
- Time window perturbation

**异构回归器集成**:
```
Regressor 1: Distribution-based (ZILN-style)
Regressor 2: Log-transformed Regression
Regressor 3: Binned Classification

Final Prediction: Ensemble(f₁, f₂, f₃)
```

**集成策略**:
- Simple average
- Weighted average (validation performance)
- Stack learning (meta-learner)

**待回答问题**:
- [ ] 哪种数据增强最有效？
- [ ] 对比学习的 temperature parameter τ如何调优？
- [ ] 异构回归器的最优集成方式？

## 第二阶段：数据准备 (Week 1-2)

### 2.1 Taobao 数据处理流程

```python
# 伪代码示例
def prepare_ltv_dataset():
    # 1. 加载原始数据
    df = pd.read_csv('UserBehavior.csv')
    
    # 2. 定义时间窗口
    # 使用前 7 天作为训练，第 8-9 天作为测试
    
    # 3. 构建用户 - 物品交互序列
    user_sequences = group_by_user(df)
    
    # 4. 定义 LTV 标签
    # 方案 1: 未来 7 天购买次数
    # 方案 2: 未来 7 天是否购买 (binary)
    # 方案 3: 构建虚拟金额 (buy=1, cart=0.5, click=0.1, pv=0)
    
    # 5. 特征工程
    features = extract_features(user_sequences)
    # - 用户历史行为统计
    # - 时间特征
    # - 物品特征
    # - 交叉特征
    
    # 6. 划分 train/val/test
    return train_data, val_data, test_data
```

### 2.2 特征设计

**用户侧特征**:
- 历史 pv/click/cart/buy 次数 (不同时间窗口)
- 首次/最近一次行为时间
- 行为多样性 (unique items)
- 活跃天数

**物品侧特征**:
- 物品热度 (全局 pv/click 数)
- 物品转化率
- 物品类别 (如果有)

**交叉特征**:
- 用户 - 物品交互历史
- 会话内位置

### 2.3 数据统计分析

需要完成的分析:
- [ ] LTV 分布可视化 (直方图、CDF)
- [ ] 零膨胀比例统计
- [ ] 长尾程度量化 (Gini coefficient)
- [ ] 不同时间窗口 LTV 相关性
- [ ] 特征重要性初步分析

## 第三阶段：基线实现 (Week 2)

### 3.1 基线模型列表

1. **Linear Regression**: 最简单的回归基线
2. **XGBoost**: Tree-based SOTA 基线
3. **Standard DNN**: 深度学习基线
4. **Two-stage Model**: 
   - Stage 1: 付费概率预测 (Logistic Regression)
   - Stage 2: 付费金额预测 (Regression, only on payers)

### 3.2 评估指标实现

```python
def evaluate_ltv_model(y_true, y_pred, prob_pred=None):
    metrics = {}
    
    # 1. AUC (for payment prediction)
    metrics['auc'] = roc_auc_score((y_true > 0).astype(int), prob_pred)
    
    # 2. PCOC (Percentage of Cumulative Ordered Coverage)
    metrics['poc_10'] = calculate_pcoc(y_true, y_pred, top_k=0.1)
    metrics['poc_20'] = calculate_pcoc(y_true, y_pred, top_k=0.2)
    metrics['poc_30'] = calculate_pcoc(y_true, y_pred, top_k=0.3)
    
    # 3. Regression metrics (on full data)
    metrics['rmse'] = mean_squared_error(y_true, y_pred, squared=True) ** 0.5
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # 4. Regression metrics (on payers only)
    payer_mask = y_true > 0
    if payer_mask.sum() > 0:
        metrics['rmse_payers'] = mean_squared_error(y_true[payer_mask], y_pred[payer_mask], squared=True) ** 0.5
    
    return metrics
```

## 第四阶段：方法实现 (Week 3-6)

### 实现顺序

1. **Week 3**: ZILN (最基础的方法)
2. **Week 4**: ODMN + MDME (多任务 + 分桶)
3. **Week 5**: ExpLTV (MoE 架构)
4. **Week 6**: CMLTV (对比学习 + 集成)

### 代码组织结构

```
ltv_optimize/
├── data/
│   ├── __init__.py
│   ├── preprocess.py          # 数据预处理
│   ├── dataset.py             # PyTorch Dataset
│   └── feature_engineering.py # 特征工程
├── models/
│   ├── __init__.py
│   ├── baseline/
│   │   ├── __init__.py
│   │   ├── linear.py
│   │   ├── xgboost.py
│   │   ├── dnn.py
│   │   └── two_stage.py
│   ├── ziln.py
│   ├── odmn_mdme.py
│   ├── expltv.py
│   └── cmltv.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── loss_functions.py
│   └── utils.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   └── analyzer.py
├── experiments/
│   ├── config.yaml
│   ├── run_baseline.sh
│   ├── run_ziln.sh
│   └── ...
└── scripts/
    ├── analyze_data.py
    ├── visualize_results.py
    └── generate_report.py
```

## 第五阶段：实验与分析 (Week 7-8)

### 5.1 实验矩阵

| 模型 | 数据集 | 随机种子 | 重复次数 |
|------|--------|----------|----------|
| Baselines | Taobao | [42, 123, 456] | 3 |
| ZILN | Taobao | [42, 123, 456] | 3 |
| ODMN+MDME | Taobao | [42, 123, 456] | 3 |
| ExpLTV | Taobao | [42, 123, 456] | 3 |
| CMLTV | Taobao | [42, 123, 456] | 3 |

### 5.2 消融研究

**ZILN**:
- 移除 zero-inflation 成分 → 纯 LogNormal
- 联合训练 vs 两阶段训练

**ODMN**:
- 单时间框架 vs 多时间框架
- 有/无 ordered constraint

**ExpLTV**:
- 专家数量：2 vs 3 vs 5
- 有/无 whale detection pre-training

**CMLTV**:
- 有/无 contrastive learning
- 不同回归器组合
- 不同集成策略

### 5.3 可视化

- 学习曲线 (train/val loss)
- 预测分布 vs 真实分布
- Error analysis (哪些用户预测不准)
- Feature importance
- t-SNE 可视化表示学习质量

## 第六阶段：报告撰写 (Week 8)

### 6.1 实验报告结构

1. **引言**: LTV 预测的重要性与挑战
2. **相关工作**: 四种方法的理论背景
3. **方法论**: 详细实现细节
4. **实验设置**: 数据集、基线、指标
5. **结果分析**: 主实验 + 消融研究
6. **案例研究**: 具体用户/场景分析
7. **结论与展望**: 总结 + 未来方向

### 6.2 预期贡献

1. **系统性的基准测试**: 首个 unified benchmark
2. **开源实现**: 所有方法的参考实现
3. **实证洞察**: 什么场景用什么方法
4. **改进建议**: 基于实验的发现

## 立即行动项

### 今日完成 ✅
- [x] 创建项目目录
- [x] 编写 README.md
- [x] 编写 experiment_report.md
- [x] 选择数据集 (Taobao)
- [x] 制定详细实施计划

### 本周目标
- [ ] 深入阅读 4 篇论文，做笔记
- [ ] 完成 Taobao 数据探索性分析
- [ ] 搭建基础代码框架
- [ ] 实现数据加载和预处理管道

### 下周目标
- [ ] 实现所有基线模型
- [ ] 建立完整的训练/评估流程
- [ ] 开始实现 ZILN

## 资源链接

**论文 PDF** (建议下载到本地):
- ZILN: https://arxiv.org/pdf/1912.07753
- ODMN&MDME: https://arxiv.org/pdf/2208.13358
- ExpLTV: https://arxiv.org/pdf/2308.12729
- CMLTV: https://arxiv.org/pdf/2306.14400

**相关代码库** (参考实现):
- TensorFlow Models Garden
- Alibaba MMEngine
- Various GitHub reproductions (需甄别质量)

**工具链**:
- PyTorch 2.0+
- Python 3.10+
- Jupyter Lab for exploration
- Weights & Biases for experiment tracking (optional)

---

*创建时间：2026-04-02*
*最后更新：2026-04-02 20:15*
*状态：计划已制定，等待开始执行*
