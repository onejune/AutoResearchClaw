# LTV Optimization - 用户生命周期价值优化研究

广告推荐中的用户 LTV（生命周期价值）建模旨在预测用户未来可能带来的收益，以优化广告投放策略。

## 研究背景

LTV 预估存在三大挑战：
1. **数据稀疏** - 大量用户没有付费行为
2. **零膨胀 (zero-inflated)** - 约 32% 用户从不付费
3. **长尾分布 (long-tailed)** - 少数大 R 用户贡献大部分收益

## 研究方向

系统性地复现和对比四种业界代表性 LTV 建模方案：

| # | 方法 | 公司 | 年份 | 核心思想 |
|---|------|------|------|----------|
| 1 | ZILN | Google | 2019 | 零膨胀对数正态分布 |
| 2 | ODMN+MDME | Kuaishou | 2022 | 多时间框架 + 分桶采样 |
| 3 | ExpLTV | Tencent | 2023 | MoE + 大 R 检测 |
| 4 | CMLTV | Huawei | 2023 | 对比学习 + 异构集成 |

## 数据集

**Taobao UserBehavior Dataset** (自主选择，不遵循 IVR 数据集规范)

- **路径**: `/mnt/data/oss_wanjun/pai_work/open_research/dataset/taobao/UserBehavior.csv`
- **规模**: ~1 亿条记录，987,994 用户，4,162,024 商品
- **特征**: user_id, item_id, category_id, behavior_type (pv/cart/fav/buy), timestamp
- **零膨胀率**: 31.94% (非购买用户)
- **长尾程度**: Top 1% 用户贡献 6.61% 购买

## 目录结构

```
ltv_optimize/
├── src/
│   ├── data/           # 数据加载和预处理
│   ├── models/         # 模型实现 (baseline, ziln, odmn, expltv, cmltv)
│   ├── train/          # 训练器
│   └── utils/          # 工具函数
├── experiments/        # 实验配置 (每个实验一个目录)
│   ├── exp001_baseline/
│   ├── exp002_ziln/
│   ├── exp003_odmn_mdme/
│   ├── exp004_expltv/
│   └── exp005_cmltv/
├── results/           # 实验结果
├── scripts/           # 实验脚本
├── configs/           # 配置文件
├── experiment_report.md      # 实验报告
├── experiments_config.yaml   # 实验调度配置
└── README.md
```

## 快速开始

```bash
# 运行 baseline 实验
python scripts/run_exp001_baseline.py

# 运行 ZILN 实验
python scripts/run_exp002_ziln.py

# 查看实验结果
cat results/exp001_baseline/results.json
```

## 评估指标

- **AUC**: 付费预测的区分能力
- **PCOC@K**: Top K%用户的 LTV 覆盖率 (K=10%, 20%, 30%)
  - 全局维度
  - business_type 维度
- **RMSE/MAE**: 回归精度
- **Log-Likelihood**: 概率模型拟合优度

## 实验结果概览

| 实验 | 方法 | AUC | PCOC@10 | 状态 |
|------|------|-----|---------|------|
| exp001 | Baseline (XGBoost) | - | - | ⏳ 待运行 |
| exp002 | ZILN | - | - | 📋 待启动 |
| exp003 | ODMN+MDME | - | - | 📋 待启动 |
| exp004 | ExpLTV | - | - | 📋 待启动 |
| exp005 | CMLTV | - | - | 📋 待启动 |

详见 [experiment_report.md](./experiment_report.md)

## 参考资料

1. **ZILN**: https://arxiv.org/pdf/1912.07753
2. **ODMN&MDME**: https://arxiv.org/pdf/2208.13358
3. **ExpLTV**: https://arxiv.org/pdf/2308.12729
4. **CMLTV**: https://arxiv.org/pdf/2306.14400

## Git 同步

本项目代码同步到：`/mnt/workspace/git_project/AutoResearchClaw/ltv_optimize/`
