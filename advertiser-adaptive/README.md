# Advertiser Adaptive — DSP 分广告主自适应建模

> P9 研究方向 | 数据集：ivr_sample_v16 | 目标会议：KDD / WWW / RecSys

## 背景

DSP 电商场景下，不同广告主（shein / aliexpress / shopee / lazada）的转化行为差异显著，全局单模型无法充分捕捉各广告主特性。本项目用多场景建模思想，分广告主自适应学习，提升各广告主分组 AUC。

## 方案

| 模型 | 说明 | 状态 |
|------|------|------|
| Baseline (DNN) | 全局单模型，不区分 domain | ✅ 完成 |
| Shared-Bottom | 共享 bottom + 各 domain tower | 🔄 重跑中 |
| MMoE | 多 expert + domain gate | 🔄 重跑中 |
| PLE | CGC 结构，专属+共享 expert | 🔄 重跑中 |
| **STAR** | 共享参数 ⊙ 场景参数，核心方案 | 🔄 重跑中 |

---

## 数据集

### 来源
- **数据集名称**：ivr_sample_v16（内部 DSP IVR 广告数据）
- **原始路径**：`/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet/YYYY-MM-DD/`
- **日期范围**：2025-11-01 ~ 2026-03-14（可用）

### 当前实验窗口
| 集合 | 日期范围 | 行数 | 正样本数 | 正样本率 |
|------|---------|------|---------|---------|
| 训练集 | 2026-03-01 ~ 2026-03-14 | 22,412,695 | 1,826,480 | 8.15% |
| 验证集 | 2026-03-15 ~ 2026-03-17 | 4,786,249 | 371,341 | 7.76% |

### 广告主过滤与 Domain 映射
只保留以下 4 类广告主（business_type 前缀匹配）：

| 广告主 | business_type 前缀 | domain id | 训练集占比 |
|--------|-------------------|-----------|-----------|
| shein | `shein` | 0 | ~3% |
| aliexpress | `ae` | 1 | ~35% |
| shopee | `shopee` | 2 | ~42% |
| lazada | `lazada` | 3 | ~20% |

### 采样策略
- **正样本（label=1）**：全部保留
- **负样本（label=0）**：
  - shein 或 `objective_type=SALES_WEBSITE`：随机保留 **1%**（seed=42）
  - 其他广告主：随机保留 **10%**（seed=42）
- **原因**：shein 负样本量远大于其他广告主，1% 采样使各 domain 样本量更均衡

### 特征
- **特征文件**：`combine_schema`（每行一个特征名）
- **特征数量**：186 个稀疏特征
- **特征类型**：全部视为稀疏特征，转字符串后用 Python `hash()` 映射到 embedding index
- **词表大小**：100,000（hash 取模，index 0 保留给缺失值）
- **缺失值处理**：填充为字符串 `"none"`，映射到 index 0

### 数据预处理流程
```
原始 parquet（ivr_sample_v16）
    ↓ PySpark 读取多天数据
    ↓ 广告主白名单过滤（shein/ae/shopee/lazada）
    ↓ 差异化负采样（shein 1%，其他 10%）
    ↓ 添加 domain_indicator 字段
    ↓ 保存为 dataset/train_XXXX/data.parquet
    ↓ PyTorch IVRDataset 加载，字符串 hash → embedding index
    ↓ 训练
```

### 预处理脚本
```bash
python scripts/prepare_dataset.py \
  --start_date 2026-03-01 \
  --end_date 2026-03-14 \
  --val_date 2026-03-15,2026-03-16,2026-03-17 \
  --output_dir ./dataset \
  --schema_path ./combine_schema \
  --data_path /mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet
```

---

## 快速开始

```bash
# 1. 数据预处理（生成 dataset/，约 40 min）
python scripts/prepare_dataset.py \
  --start_date 2026-03-01 \
  --end_date 2026-03-14 \
  --val_date 2026-03-15,2026-03-16,2026-03-17

# 2. 训练单个实验
python scripts/train.py --conf conf/experiments/exp_005_star.yaml

# 3. 串行跑 exp_002~005
python scripts/run_exp_002_005.py
```

---

## 实验结果

见 `experiments/leaderboard.json`（自动更新）和 `experiments/experiment_report.md`（完整报告）。

---

## 目录结构

```
exp_advertiser_adaptive/
├── README.md               # 本文档
├── DESIGN.md               # 详细设计文档
├── combine_schema          # 特征列表（186个）
├── feat_conf               # 特征配置原始文件
├── conf/
│   ├── base.yaml           # 基础训练配置
│   └── experiments/        # 各实验 yaml 配置
│       ├── exp_001_baseline.yaml
│       ├── exp_002_shared_bottom.yaml
│       ├── exp_003_mmoe.yaml
│       ├── exp_004_ple.yaml
│       └── exp_005_star.yaml
├── src/
│   ├── data/
│   │   ├── dataset.py      # IVRDataset（PyTorch）
│   │   └── spark_loader.py # PySpark 数据加载 & 采样
│   ├── models/             # 模型实现（Baseline/SharedBottom/MMoE/PLE/STAR）
│   ├── trainer.py          # 训练循环
│   └── utils/
├── scripts/
│   ├── prepare_dataset.py  # 数据预处理
│   ├── train.py            # 单实验训练入口
│   ├── run_experiments.sh  # 并行批量运行
│   └── run_exp_002_005.py  # 串行跑 exp_002~005 + 飞书通知
├── dataset/                # 预处理数据缓存（gitignore）
└── experiments/            # 实验结果（leaderboard + metrics）
    ├── leaderboard.json
    ├── experiment_report.md
    └── exp_00X_*/
        └── metrics.json
```

---

## 参考

- [STAR: One Model to Serve All (CIKM 2021)](https://dl.acm.org/doi/abs/10.1145/3459637.3481941)
- [PLE: Progressive Layered Extraction (RecSys 2020)](https://dl.acm.org/doi/10.1145/3383313.3412236)
- [MMoE: Modeling Task Relationships (KDD 2018)](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-)
- [Scenario-Wise-Rec Benchmark](https://github.com/Xiaopengli1/Scenario-Wise-Rec)
