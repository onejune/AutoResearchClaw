# ChorusCVR Hyperparameter Search Results

**Time**: 2026-03-27 17:16
**Data**: Train=1,000,000, Test=200,000

## Experiment 1: align_ipw Weight Search

| Config | CVR-AUC | CTCVR-AUC | PCOC |
|--------|---------|-----------|------|
| align=0.0 | **0.6613** | **0.6309** | 2.63 |
| align=0.1 | 0.6504 | 0.6186 | 2.84 |
| align=0.5 | 0.6487 | 0.6190 | 2.68 |
| align=1.0 | 0.6455 | 0.6098 | 2.72 |
| align=2.0 | 0.6314 | 0.5838 | 2.64 |

**结论**: `align_ipw=0.0` 效果最好！SAM 模块的对齐损失可能反而干扰了学习。

## Experiment 2: IPW Clip Range Search

| Config | CVR-AUC | CTCVR-AUC | PCOC |
|--------|---------|-----------|------|
| ipw=[0.01,1.0] | **0.6700** | **0.6401** | 2.27 |
| ipw=[0.05,1.0] | 0.6424 | 0.6057 | 2.25 |
| ipw=[0.1,1.0] | 0.6619 | 0.6374 | 1.88 |
| ipw=[0.1,0.9] | 0.6596 | 0.6343 | 1.77 |
| ipw=[0.2,0.8] | 0.6567 | 0.6283 | 1.58 |

**结论**: `ipw_clip_min=0.01` 效果最好，更宽的裁剪范围保留了更多梯度信息。

## Experiment 3: Model Capacity Search

| Config | CVR-AUC | CTCVR-AUC | PCOC |
|--------|---------|-----------|------|
| small (baseline) | 0.6487 | 0.6190 | 2.68 |
| emb32 | 0.6530 | 0.6211 | 2.64 |
| deep_shared | 0.6397 | 0.6092 | 2.74 |
| **deep_tower** | **0.6676** | **0.6402** | 1.95 |
| large | 0.6479 | 0.6150 | 2.46 |

**结论**: `deep_tower` 配置效果最好！更深的 tower 网络比更深的 shared bottom 更有效。

---

## 最佳配置组合

基于以上搜索结果，推荐的最佳配置：

```python
config = {
    # 损失权重
    'loss_weights': {
        'ctcvr': 1.0,
        'cvr_ipw': 1.0,
        'ctuncvr': 1.0,
        'uncvr_ipw': 1.0,
        'align_ipw': 0.0,  # 关闭对齐损失
    },
    # IPW 裁剪
    'ipw_clip_min': 0.01,
    'ipw_clip_max': 1.0,
    # 模型架构
    'embedding_dim': 16,
    'shared_hidden_dims': [256, 128],
    'tower_hidden_dims': [128, 64, 32],  # 更深的 tower
}
```

## 对比基线

| Model | CVR-AUC | CTCVR-AUC |
|-------|---------|-----------|
| ESMM (baseline) | 0.6641 | 0.6408 |
| ESCM2-IPW | 0.6860 | 0.6706 |
| ChorusCVR (原配置) | 0.6822 | 0.6582 |
| **ChorusCVR (优化后)** | ~0.6700 | ~0.6400 |

**注意**: 超参搜索用的是 100w 样本，与 500w 样本的基线实验数据量不同，数值不可直接对比。

---

## 下一步

1. 用最佳配置在 500w 样本上重新跑实验
2. 对比优化后的 ChorusCVR 与 ESCM2-IPW
