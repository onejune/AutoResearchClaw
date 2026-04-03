# LTV Optimization Research - Initialization Complete ✅

## 项目启动总结

**时间**: 2026-04-02 20:10-20:30  
**状态**: 初始化完成，数据探索完成，准备进入基线实现阶段

---

## 📋 已完成工作

### 1. 项目结构搭建 ✅
```
/mnt/workspace/open_research/autoresearch/ltv_optimize/
├── README.md                      # 项目概述
├── experiment_report.md           # 实验报告 (已更新数据发现)
├── implementation_plan.md         # 详细实施计划
├── PROJECT_SUMMARY.md             # 项目摘要
├── research_task.md              # 研究任务定义
├── config.arc.yaml               # AutoResearchClaw 配置
├── 研究方向.txt                  # 原始研究方向
├── data/                         # 数据处理模块
├── models/                       # 模型实现目录
│   ├── baseline/
│   ├── ziln/
│   ├── odmn_mdme/
│   ├── expltv/
│   └── cmltv/
├── training/                     # 训练代码
├── evaluation/                   # 评估指标
├── experiments/                  # 实验管理
├── scripts/
│   └── explore_data.py          # 数据探索脚本
└── logs/
    ├── data_exploration.log     # 数据探索日志
    └── researchclaw_log.txt     # AutoResearchClaw 日志
```

### 2. 数据集深度分析 ✅

**选定**: Taobao UserBehavior Dataset

**核心发现**:

| 指标 | 数值 | 意义 |
|------|------|------|
| 总交互数 | 1 亿 + | 数据丰富，适合深度学习 |
| 用户数 | 987,994 | 大规模用户群体 |
| **零膨胀率** | **31.94%** | **显著，适合 ZILN/MDME** |
| Top 1% 贡献 | 6.61% | 长尾存在但温和，适合 ExpLTV |
| 购买用户比例 | 68.06% | 较高转化率 |
| 平均购买次数 | 3.00 | 中等活跃度 |
| 最大购买次数 | 262 | 存在超级大 R |

**行为漏斗**:
```
PV (99.6%) → Cart (75.1%) → Buy (91.0% of Cart)
```

**关键洞察**:
1. ✅ **零膨胀显著** (32% 非购买者) → ZILN 方法高度相关
2. ✅ **长尾分布** (Top 1% 占 6.61%) → ExpLTV 的大 R 检测有价值
3. ✅ **多行为类型** (pv/cart/fav/buy) → CMLTV 的多视角学习可行
4. ✅ **数据量大** → 支持复杂的深度学习模型
5. ⚠️ **时间戳异常** → 需要数据清洗和窗口定义

### 3. 研究方法论设计 ✅

**四种方法的适用性分析**:

| 方法 | 适用度 | 理由 |
|------|--------|------|
| **ZILN** | ⭐⭐⭐⭐⭐ | 32% 零膨胀率是典型应用场景 |
| **ODMN+MDME** | ⭐⭐⭐⭐ | 可建模 7 日/14 日/30 日 LTV 依赖；分桶处理长尾 |
| **ExpLTV** | ⭐⭐⭐⭐ | Top 1% 用户可定义为"大 R"；MoE 架构灵活 |
| **CMLTV** | ⭐⭐⭐⭐ | 多行为类型提供天然的多视角；对比学习提升鲁棒性 |

### 4. 工具链准备 ✅

- **AutoResearchClaw**: 已安装，配置文件就绪
- **Python 环境**: PyTorch, Pandas, NumPy, Matplotlib, Seaborn 可用
- **数据访问**: Taobao 数据集路径已确认

---

## 🎯 下一步行动计划

### 立即开始 (今天 - 明天)

#### Step 1: 数据预处理管道 🔨
```python
# 需要实现的功能
- 加载原始 CSV
- 时间戳校准和窗口划分 (train/val/test)
- 用户 - 物品交互序列构建
- 特征工程 (用户统计、时间特征、物品特征)
- LTV 标签生成 (未来窗口购买次数)
- 数据保存为高效格式 (parquet/hdf5)
```

#### Step 2: 基线模型实现 📊
```python
# Baseline 优先级
1. Linear Regression (快速验证 pipeline)
2. XGBoost (strong tabular baseline)
3. Two-stage Model (logistic + regression)
4. Simple DNN (deep learning baseline)
```

#### Step 3: 评估框架搭建 📏
```python
# 核心指标
- AUC (付费预测)
- PCOC@K (排序质量，K=10%, 20%, 30%)
- RMSE/MAE (回归精度)
- Log-Likelihood (概率模型)
```

### 本周目标

- [ ] 完成数据预处理管道
- [ ] 实现 4 个基线模型
- [ ] 建立完整的训练/评估流程
- [ ] 产出第一批基线结果
- [ ] 开始 ZILN 模型实现

### 本月目标

- [ ] 完成 ZILN 实现和实验
- [ ] 完成 ODMN+MDME 实现和实验
- [ ] 开始 ExpLTV 和 CMLTV 实现
- [ ] 初步对比分析报告

---

## 📊 数据洞察对研究的指导

### ZILN 实施要点
- 零膨胀率 32% → π参数很关键
- 购买次数范围 1-262 → LogNormal 拟合合理
- 建议：先用购买次数，后续尝试虚拟价值

### ODMN 实施要点
- 可定义 7 日/14 日/30 日累积购买次数
- 天然满足有序约束 (LTV_7d ≤ LTV_14d ≤ LTV_30d)
- 建议：先实现单时间窗口，再扩展多时间窗口

### MDME 实施要点
- 分桶策略建议:
  - Bucket 0: 0 购买 (32%)
  - Bucket 1: 1-2 购买 (57%)
  - Bucket 2: 3-5 购买 (9%)
  - Bucket 3: 6-10 购买 (2%)
  - Bucket 4: 11+ 购买 (<1%)
- 建议：按分位数 vs 固定阈值做对比实验

### ExpLTV 实施要点
- 大 R 定义: Top 1% (购买次数≥14) 或 Top 5%
- 专家配置: Whale Expert, Regular Expert, New User Expert
- 建议：先硬规则分群，再学习软门控

### CMLTV 实施要点
- 多视角定义:
  - View 1: 用户历史行为统计
  - View 2: 会话级特征
  - View 3: 物品侧特征
- 数据增强：Feature dropout, sample augmentation
- 建议：先从简单 contrastive loss 开始

---

## 🚨 注意事项

1. **数据划分**: 必须按时间划分，不能用随机划分 (防止数据泄露)
2. **冷启动用户**: 考虑如何处理测试集的新用户
3. **负采样**: 如果构造 user-item 对，需要合理负采样
4. **计算资源**: 深度学习模型需要 GPU，提前规划
5. **实验追踪**: 建议使用 Weights & Biases 或 MLflow

---

## 💡 研究创新点机会

基于数据特点，可能的创新方向:

1. **行为序列融合**: 结合 pv/cart/fav/buy 的序列模式
2. **动态大 R 定义**: 在线学习 evolving whale threshold
3. **跨类别迁移**: 利用 category_id 做 multi-task learning
4. **时间感知对比学习**: 考虑时间衰减的 contrastive learning
5. **解释性 LTV**: 不仅预测值，还给出驱动因素

---

## 📞 沟通与协作

**进度汇报**: 
- 每日：关键里程碑完成时通知
- 每周：周日晚总结本周进展和下周计划
- 遇到问题：立即提出，不卡壳

**决策点**(需要确认):
1. LTV 定义方案 (购买次数 vs 虚拟价值)?
2. 时间窗口选择 (7 日/14 日/30 日)?
3. 数据划分比例 (train/val/test)?
4. 优先实现哪个方法 (建议 ZILN 起步)?

---

## 📚 参考资料速查

**论文 PDF 链接**:
- ZILN: https://arxiv.org/pdf/1912.07753
- ODMN&MDME: https://arxiv.org/pdf/2208.13358  
- ExpLTV: https://arxiv.org/pdf/2308.12729
- CMLTV: https://arxiv.org/pdf/2306.14400

**关键公式速记**:
- ZILN Loss: `-Σ[I(y=0)*log(π) + I(y>0)*log((1-π)*fLN(y|μ,σ))]`
- PCOC@K: `Top K%用户的实际 LTV 总和 / 所有用户 LTV 总和`
- Contrastive Loss: `-log(exp(sim(z1,z2)/τ) / Σexp(sim(z1,z_neg)/τ))`

---

**状态**: ✅ 准备就绪，等待开始编码实现  
**最后更新**: 2026-04-02 20:30  
**下次检查点**: 数据预处理管道完成 (预计 1-2 天)
