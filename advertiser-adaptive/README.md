# Advertiser Adaptive — DSP 分广告主自适应建模

> P9 研究方向 | 数据集：ivr_sample_v16 | 目标会议：KDD / WWW / RecSys

## 背景

DSP 电商场景下，不同广告主（shein / aliexpress / shopee / lazada）的转化行为差异显著，全局单模型无法充分捕捉各广告主特性。本项目用多场景建模思想，分广告主自适应学习，提升各广告主分组 AUC。

## 方案

| 模型 | 说明 | 状态 |
|------|------|------|
| Baseline (DNN) | 全局单模型，不区分 domain | 🔄 开发中 |
| Shared-Bottom | 共享 bottom + 各 domain tower | 🔄 开发中 |
| MMoE | 多 expert + domain gate | 🔄 开发中 |
| PLE | CGC 结构，专属+共享 expert | 🔄 开发中 |
| **STAR** | 共享参数 ⊙ 场景参数，核心方案 | 🔄 开发中 |

## 数据

- **数据集**：ivr_sample_v16（内部 DSP IVR 数据）
- **路径**：`/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet/part=YYYY-MM-DD/`
- **训练**：7天训练 + 1天验证
- **采样**：正样本全保留；shein 负样本 1%，其他 10%
- **特征**：186 个单特征（见 `combine_schema`）
- **Domain 映射**：shein→0, ae*→1, shopee*→2, lazada*→3

## 快速开始

```bash
# 1. 数据预处理（生成 dataset/）
python scripts/prepare_dataset.py \
  --start_date 2025-01-01 \
  --end_date 2025-01-07 \
  --val_date 2025-01-08 \
  --output_dir ./dataset

# 2. 训练单个实验
python scripts/train.py --conf conf/experiments/exp_005_star.yaml --exp_name exp_005_star

# 3. 批量并行实验
bash scripts/run_experiments.sh
```

## 实验结果

见 `experiments/leaderboard.json`（自动更新）

## 目录结构

```
advertiser-adaptive/
├── conf/                   # 配置文件
│   ├── base.yaml
│   └── experiments/        # 各实验配置
├── src/                    # 核心代码
│   ├── data/               # 数据加载
│   ├── models/             # 模型实现
│   ├── trainer.py
│   └── utils/
├── scripts/                # 运行脚本
├── dataset/                # 预处理数据（gitignore）
├── experiments/            # 实验结果（gitignore）
├── combine_schema          # 特征列表
└── DESIGN.md               # 详细设计文档
```

## 参考

- [STAR: One Model to Serve All (CIKM 2021)](https://dl.acm.org/doi/abs/10.1145/3459637.3481941)
- [PLE: Progressive Layered Extraction (RecSys 2020)](https://dl.acm.org/doi/10.1145/3383313.3412236)
- [MMoE: Modeling Task Relationships (KDD 2018)](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-)
- [Scenario-Wise-Rec Benchmark](https://github.com/Xiaopengli1/Scenario-Wise-Rec)
