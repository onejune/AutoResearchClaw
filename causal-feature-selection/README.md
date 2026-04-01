# 因果特征选择与 OOD 泛化 for CTR 预估

**研究问题**: 如何在 CTR 预估中提升模型在分布外 (OOD) 场景下的泛化能力？  
**数据集**: IVR Sample v16 CTCVR（304 万训练，20 万测试）  
**启动日期**: 2026-04-01  
**状态**: Phase 1-3 完成

---

## 核心结论

| 方法 | 整体 AUC | bt_16 AUC | 结论 |
|------|----------|-----------|------|
| Baseline (WideDeep) | **0.7545** | 0.5110 | 对照组 |
| IRM (λ=100) | 0.7317 | 0.5237 | ❌ 无效，AUC 下降 |
| 去虚假特征 | 0.7487 | 0.4886 | ❌ 无效，bt_16 更差 |
| 只因果特征 | 0.6786 | 0.5243 | ❌ 无效，大幅下降 |
| **DANN (λ=0.1)** | 0.7339 | **0.5419** | ✅ bt_16 提升 3 千分点 |

**关键发现**：
1. IRM 在本场景下无效（各环境梯度本身一致，惩罚项 ~0.0003）
2. 特征子集选择无效（因果特征集太小，去虚假特征反而损害性能）
3. DANN 是唯一有效方法，bt_16 AUC 从 0.49→0.54

---

## 目录结构

```
causal_feature_selection/
├── src/
│   ├── data/
│   │   └── loader.py             # IVR v16 数据加载器
│   ├── models/
│   │   └── baseline.py           # WideDeep 模型
│   ├── methods/
│   │   ├── feature_importance.py # 特征重要性（EmbNorm + 梯度）
│   │   ├── irm.py                # IRM 多环境训练
│   │   └── dann.py               # DANN 域对抗训练
│   └── train/
│       └── trainer.py            # 统一训练接口
├── scripts/
│   ├── run_baseline.py           # exp001: 基线实验
│   ├── run_feature_importance.py # exp002: 特征重要性分析
│   ├── run_exp003_irm.py         # exp003: IRM 消融
│   ├── run_exp004_feature_subset.py # exp004: 特征子集选择
│   └── run_exp005_dann.py        # exp005: DANN 域对抗
├── experiments/
│   ├── README.md                 # 实验索引
│   ├── exp001_baseline/
│   ├── exp002_feature_importance/
│   ├── exp003_irm/
│   ├── exp004_feature_subset/
│   └── exp005_dann/
├── results/                      # 模型权重 & 原始输出
├── configs/                      # 实验配置
├── EXPERIMENT_SUMMARY.md         # 所有实验汇总对比
├── experiment_report.md          # 详细实验报告
├── DATASET.md                    # 数据集说明
└── requirements.txt
```

---

## 快速上手

### 环境准备

```bash
pip install -r requirements.txt
```

### 数据路径

```
/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/
```

### 运行实验

```bash
cd /mnt/workspace/open_research/autoresearch/causal_feature_selection

# 基线
python scripts/run_baseline.py

# 特征重要性分析
python scripts/run_feature_importance.py

# IRM 消融 (λ = 0/1/10/100)
python scripts/run_exp003_irm.py

# 特征子集选择
python scripts/run_exp004_feature_subset.py

# DANN 域对抗 (λ = 0/0.1/1/10)
python scripts/run_exp005_dann.py
```

---

## 实验设置

### 数据划分

| 集合 | BT | 样本数 | 说明 |
|------|-----|--------|------|
| 训练集 | 0,1,2,3,4,5,6,11 | 3,045,806 | 源域 |
| 测试集 | 7,8,10,13,16 | 204,951 | OOD 目标域 |

### 测试集各 BT 分布

| BT | 样本数 | CTR | 分布漂移 |
|----|--------|-----|----------|
| bt_7 | 15,031 | 28.6% | 中等 |
| bt_8 | 119,026 | 24.6% | 中等 |
| bt_10 | 50,987 | 48.3% | 小 |
| bt_13 | 12,343 | 37.3% | 小 |
| **bt_16** | 7,564 | **66.1%** | **极大** |

### 模型配置

```python
WideDeepBaseline(
    embedding_size=8,
    dnn_hidden_units=[1024, 512, 256, 128],
    dropout=0.3,
    lr=5e-5,
    epochs=1  # 广告流式数据，只跑 1 epoch
)
```

---

## 参考文献

- **IRM**: Arjovsky et al., "Invariant Risk Minimization", arXiv 2019
- **DANN**: Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016
- **WideDeep**: Cheng et al., "Wide & Deep Learning for Recommender Systems", RecSys 2016
