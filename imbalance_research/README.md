# Focal Loss 类别不均衡研究

## 项目概述

专注于 CTR/CVR 预估场景下的类别不均衡问题，重点探索 Focal Loss 及其变体的应用效果。

## 快速开始

```bash
# 1. 查看可用实验
python run.py --list

# 2. 运行单个实验
python run.py --exp baseline_focal

# 3. 调试模式 (小数据, 2 epochs)
python run.py --exp baseline_focal --debug

# 4. 指定 GPU
python run.py --exp baseline_focal --gpu 0

# 5. 批量实验
python run.py --batch experiments/batch.yaml

# 6. 覆盖参数
python run.py --exp baseline_focal --epochs 50 --lr 0.0001
```

## 目录结构

```
imbalance_research/
├── run.py              # 统一入口
├── core/               # 核心模块
│   ├── experiment.py   # 实验类
│   └── registry.py     # 实验注册表
├── models/             # 模型定义
│   ├── losses.py       # Focal Loss 变体
│   └── embedding_model.py
├── data/               # 数据加载
│   └── dataset.py
├── evaluation/         # 评估指标
│   └── metrics.py
├── experiments/        # 实验配置
│   └── batch.yaml      # 批量实验
└── results/            # 实验结果 (自动生成)
    └── experiments/    # 每个实验独立目录
        └── {name}_{id}/
            ├── config.json
            ├── results.json
            └── checkpoints/
```

## 已注册实验

| 实验名 | 描述 |
|--------|------|
| `baseline_bce` | MLP + BCE Loss (基准) |
| `baseline_focal` | MLP + Focal Loss (γ=2, α=0.25) |
| `focal_gamma_*` | Gamma 参数扫描 (0.5~5.0) |
| `focal_alpha_*` | Alpha 参数扫描 (0.1~0.5) |
| `balanced_focal` | 自动调整 alpha |
| `asymmetric_focal` | 正负样本不同 gamma |
| `dynamic_focal` | gamma 随训练衰减 |
| `smoothed_focal` | Label Smoothing |

## 添加新实验

编辑 `core/registry.py`:

```python
register_experiment(
    name='my_experiment',
    description='我的实验描述',
    config={
        'loss': {
            'type': 'focal',
            'gamma': 3.0,
            'alpha': 0.3,
        },
        'model': {
            'type': 'mlp',
            'hidden_dims': [512, 256, 128],
        },
        'training': {
            'epochs': 30,
            'batch_size': 512,
        },
    }
)
```

## 结果说明

每个实验完成后，结果保存在 `results/experiments/{name}_{id}/`:

- `config.json`: 完整配置
- `results.json`: 训练历史和最终指标
- `checkpoints/best.pt`: 最优模型
- `checkpoints/latest.pt`: 最新模型

## 核心指标

- **AUC**: 主要评估指标
- **PCOC**: 预测/实际 CTR 比值 (校准度，接近 1 最好)
- **LogLoss**: 交叉熵损失
- **F1/Precision/Recall**: 分类指标

---

**创建时间**: 2026-03-30  
**重构时间**: 2026-04-01  
**负责人**: 秦始皇 👑
