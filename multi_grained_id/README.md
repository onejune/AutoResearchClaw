# Multi-Grained ID Embedding

广告 ID 多粒度 Embedding 方案对比实验

## 目录结构

```
.
├── src/
│   ├── data/           # 数据加载
│   ├── models/         # 模型实现
│   ├── train/          # 训练器
│   └── utils/          # 工具（HardwareMonitor）
├── scripts/            # 实验脚本
├── results/           # 实验结果
└── experiments/        # 实验配置
```

## 快速开始

```bash
# 运行单个实验
python scripts/run_exp004_metaemb.py

# 查看结果
cat results/exp004_metaemb/results.json
```

## 实验结果

| 方法 | AUC | PCOC |
|------|-----|------|
| Baseline | 0.8459 | 1.0821 |
| MetaEmb | **0.8462** 🏆 | 1.1762 |
| DDS | 0.8459 | **0.9833** ✅ |
| Hierarchical | 0.8454 | 1.0014 |

详见 [experiment_report.md](./experiment_report.md)