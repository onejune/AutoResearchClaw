# LTV Optimization - Experiment Report

## 项目信息

- **项目名称**: P15 - LTV (Life-Time Value) Optimization in Advertising Recommendation Systems
- **开始时间**: 2026-04-02
- **当前状态**: 📋 待启动 (基线框架就绪)
- **数据集**: Taobao UserBehavior Dataset (自主选择)

---

## 数据集说明

### 数据来源
**路径**: `/mnt/data/oss_wanjun/pai_work/open_research/dataset/taobao/UserBehavior.csv`

**注意**: 本项目为 LTV 研究，根据 research_notes.md 允许自主选择数据集，不使用 IVR 数据集。

### 数据统计

| 指标 | 数值 |
|------|------|
| 总交互数 | 100,150,807 |
| 用户数 | 987,994 |
| 商品数 | 4,162,024 |
| 行为类型 | pv (89.58%), cart (5.52%), fav (2.88%), buy (2.01%) |

### 关键特征

- **零膨胀率**: 31.94% (非购买用户比例)
- **长尾程度**: Top 1% 用户贡献 6.61% 购买
- **平均 LTV**: 3.00 次购买
- **最大 LTV**: 262 次购买
- **购买用户**: 672,404 (68.06%)

### LTV 定义

使用**未来 7 天内的购买次数**作为 LTV 标签

---

## 实验设计

### exp001: Baseline Models ⏳ 待运行

**目标**: 建立性能基准，验证数据管道

**模型列表**:
1. Linear Regression
2. XGBoost
3. Two-stage Model (Classification + Regression)
4. Simple DNN (1 epoch)

**评估指标**:
- AUC (付费预测)
- PCOC@10, PCOC@20, PCOC@30 (排序质量)
- RMSE, MAE (回归精度)

**运行命令**:
```bash
python scripts/run_exp001_baseline.py
```

**预期输出**: `results/exp001_baseline/results.json`

---

### exp002: ZILN 📋 待启动

**论文**: Zhang et al. "Predicting Player Lifetime Value with Zero-Inflated Log-Normal." Google, 2019.  
**链接**: https://arxiv.org/pdf/1912.07753

**核心思想**: 
- 零膨胀对数正态分布建模
- DNN 输出三个参数：π (zero prob), μ, σ (lognormal params)
- 损失函数：Negative Log-Likelihood

**适用场景**: 高零膨胀率数据 (本项目 32%，非常适合)

---

### exp003: ODMN + MDME 📋 待启动

**论文**: Yu et al. "ODMN: Ordered Deep Multi-timeframe Network for LTV Prediction." Kuaishou, 2022.  
**链接**: https://arxiv.org/pdf/2208.13358

**ODMN**: 多时间框架有序依赖建模 (7d/14d/30d LTV)  
**MDME**: 分桶采样处理极端不平衡

**适用场景**: 需要多时间窗口预测 + 长尾分布

---

### exp004: ExpLTV 📋 待启动

**论文**: Liu et al. "ExpLTV: A New Model for LTV Prediction with Expert Network." Tencent, 2023.  
**链接**: https://arxiv.org/pdf/2308.12729

**核心创新**: 
- 大 R 用户检测作为门控网络
- Mixture-of-Experts 架构
- 不同用户类型路由到专属专家

**适用场景**: 用户异质性强，存在明显的大 R 群体

---

### exp005: CMLTV 📋 待启动

**论文**: Chen et al. "CMLTV: Contrastive Multi-view Learning for LTV Prediction." Huawei, 2023.  
**链接**: https://arxiv.org/pdf/2306.14400

**核心思想**:
- 对比学习提升表示鲁棒性
- 多视角数据增强
- 异构回归器集成

**适用场景**: 需要鲁棒性，多源特征融合

---

## 实验结果

### 基线模型结果 (exp001)

*待运行后填充*

| Model | AUC | PCOC@10 | PCOC@20 | PCOC@30 | RMSE | MAE |
|-------|-----|---------|---------|---------|------|-----|
| LinearRegression | - | - | - | - | - | - |
| XGBoost | - | - | - | - | - | - |
| TwoStage | - | - | - | - | - | - |
| SimpleDNN | - | - | - | - | - | - |

### 高级模型结果

*待后续实验填充*

| Experiment | Method | AUC | PCOC@10 | RMSE | 状态 |
|------------|--------|-----|---------|------|------|
| exp002 | ZILN | - | - | - | 📋 待启动 |
| exp003 | ODMN+MDME | - | - | - | 📋 待启动 |
| exp004 | ExpLTV | - | - | - | 📋 待启动 |
| exp005 | CMLTV | - | - | - | 📋 待启动 |

---

## 实验进度

- [x] 项目初始化
- [x] 数据探索分析
- [x] 基线框架搭建
- [x] 实验目录结构创建
- [x] research_list.md 更新
- [ ] **exp001: Baseline 训练和评估**
- [ ] exp002: ZILN 实现
- [ ] exp003: ODMN+MDME 实现
- [ ] exp004: ExpLTV 实现
- [ ] exp005: CMLTV 实现
- [ ] 综合对比分析
- [ ] 研究报告撰写

---

## 关键技术点

### 数据预处理
- ✅ 按时间划分 train/val/test (防止泄露)
- ✅ 用户级特征工程完成
- ✅ LTV 标签生成逻辑确认

### 模型实现注意事项
- ⚠️ 遵循 research_notes.md: CTR/CVR 类模型只跑 1 epoch
- ⚠️ 评估必须包括全局和 business_type 维度
- ⚠️ 所有实验共用 baseline，避免重复训练

### GPU 调度
- 参考 `multi_grained_id` 项目的实验管理系统
- 使用 `experiments_config.yaml` 配置实验优先级和资源需求

---

## 参考资料

1. **ZILN**: https://arxiv.org/pdf/1912.07753
2. **ODMN & MDME**: https://arxiv.org/pdf/2208.13358
3. **ExpLTV**: https://arxiv.org/pdf/2308.12729
4. **CMLTV**: https://arxiv.org/pdf/2306.14400

---

*最后更新：2026-04-02 20:45*  
*报告状态：框架就绪，等待基线实验运行*
