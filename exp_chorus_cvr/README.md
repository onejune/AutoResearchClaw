# ChorusCVR 复现项目

复现快手 ChorusCVR 论文 (arXiv:2502.08277v2)，解决 CVR 预测中的样本选择偏差 (SSB) 问题。

## 核心结果 (500万样本)

| Model | CVR-AUC | CTCVR-AUC | vs ESMM |
|-------|---------|-----------|---------|
| ESMM | 0.6750 | 0.6509 | - |
| ESCM2-IPW | 0.6732 | 0.6589 | -0.18% / +0.80% |
| **ChorusCVR** | **0.6829** | **0.6656** | **+0.79% / +1.48%** |

## 快速开始

```bash
cd /mnt/workspace/open_research/autoresearch/exp_chorus_cvr

# 基线对比实验 (500万样本, ~10分钟)
python scripts/run_all_baselines.py --max_samples 5000000

# 超参搜索 (100万样本, ~15分钟)
python scripts/run_hyperparam_search.py

# 最佳配置验证 (500万样本)
python scripts/run_best_config.py
```

## 数据集

Ali-CCP: `/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp/`
- 训练集: 2,859,202 样本
- 测试集: 1,454,318 样本
- 特征: 19 稀疏 + 8 稠密

## 核心创新

1. **NDM (Negative sample Discrimination Module)**: 引入 unCVR 任务，区分"点击未转化"和"未点击"两类负样本
2. **SAM (Soft Alignment Module)**: CVR 和 unCVR 互监督对齐 (实验发现 align_ipw=0 效果更好)

## 关键发现

1. **align_ipw=0 效果最好**：关闭 SAM 对齐损失
2. **deep_tower 更优**：`[128,64,32]` 比浅层有效
3. **PCOC ≈ 1.09**：校准性良好

## 项目结构

```
exp_chorus_cvr/
├── src/
│   ├── models/
│   │   ├── chorus_cvr.py      # ChorusCVR 模型
│   │   └── baselines.py       # ESMM/ESCM2/DCMT/DDPO
│   └── losses/
│       ├── chorus_loss.py     # ChorusCVR 损失
│       └── baseline_losses.py # 基线损失
├── scripts/
│   ├── run_all_baselines.py   # 基线对比
│   ├── run_hyperparam_search.py # 超参搜索
│   └── run_best_config.py     # 最佳配置
├── results/                   # 实验结果
├── experiment_report.md       # 完整实验报告
└── README.md
```

## 最佳配置

```python
config = {
    'tower_hidden_dims': [128, 64, 32],  # 更深的 tower
    'loss_weights': {
        'ctcvr': 1.0, 'cvr_ipw': 1.0,
        'ctuncvr': 1.0, 'uncvr_ipw': 1.0,
        'align_ipw': 0.0,  # 关闭对齐损失
    },
}
```

## 详细结果

见 `experiment_report.md`
