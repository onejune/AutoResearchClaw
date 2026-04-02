# Bid Landscape Forecasting 🏆

预测竞价胜率 (win_rate)，解决删失数据问题。

## 📊 项目概览

- **数据集**: Synthetic Bid Landscape (50 万样本，基于 IVR Sample v16 CTCVR)
- **目标**: 预测竞价胜率 (win_rate)
- **核心发现**: 时序建模 (LSTM+Attention) 效果最佳，AUC=0.9497

## 🏆 最终排名 (Top 10)

| 排名 | 实验 | AUC | 方法类型 |
|------|------|-----|----------|
| 🥇 | exp12_deepwin | **0.9497** | LSTM+Attention |
| 🥈 | exp06_mtlsa | 0.8694 | Multi-task Learning |
| 🥉 | exp05_dlf | 0.8687 | GRU Sequence |
| 4 | exp09_conformal | 0.8655 | Conformal Prediction |
| 5 | exp10_deep_censored | 0.8649 | Deep Censored |
| 6 | exp08_quantile | 0.8627 | Quantile NN |
| 7 | exp11_quantile_forest | 0.8590 | Quantile RF |
| 8 | exp16_adversarial | 0.8570 | Counterfactual |
| 9 | exp19_tabtransformer | 0.8432 | TabTransformer |
| 10 | exp13_deephit | 0.8369 | Survival Analysis |

## 🔑 核心发现

### 1. 时序建模是王道

DeepWin (LSTM+Attention) 利用历史竞价序列，比其他方法高出 ~8% AUC。

```
输入：context_features + bid_sequence (5 个历史竞价)
  ↓
Bid Embedding (16-dim per bid)
  ↓
LSTM (2-layer, 128-hidden) → 捕捉时序依赖
  ↓
Attention → 加权重要时间步
  ↓
Fusion (context + sequence features)
  ↓
输出：win_probability
```

### 2. 数据特性限制

`true_value` 与 `win_label` 几乎无关 (r=0.0017)，导致 counterfactual inference 方法效果有限。

### 3. 简单方法仍有效

GB/LR AUC=0.83-0.84，推理<1ms，适合生产环境快速部署。

## 📁 项目结构

```
bid_landscape_forecasting/
├── analysis/           # 数据分析脚本
├── checkpoints/        # 模型检查点
├── data/              # 数据处理模块
├── experiments/       # 实验配置
├── figures/           # 可视化图表
├── logs/              # 训练日志
├── models/            # 训练好的模型
├── results/           # 实验结果 (JSON/Markdown)
├── scripts/           # 运行脚本
├── utils/             # 工具函数
├── FINAL_SUMMARY.md   # 项目总结
├── EXPERIMENT_REPORT.md  # 实验报告
└── README.md          # 本文档
```

## 🚀 快速开始

### 安装依赖

```bash
pip install torch pandas scikit-learn xgboost lightgbm
```

### 运行实验

```bash
# 运行所有实验
python -m experiments.run_all

# 运行单个实验
python -m experiments.exp12_deepwin

# 查看结果
cat results/exp12_deepwin.json
```

## 💡 工业应用建议

### 场景 1: 有历史竞价数据

**推荐**: DeepWin (LSTM+Attention)
- AUC: 0.95
- 推理时间：~1ms
- 训练时间：~30s (10 万样本)

### 场景 2: 只有当前出价信息

**推荐**: MTLSA 或 GB
- AUC: 0.83-0.87
- 推理时间：<1ms
- 训练时间：<1s

### 场景 3: 需要不确定性估计

**推荐**: Conformal Prediction (exp09)
- AUC: 0.86
- Coverage: 90%
- 提供置信区间

## 📝 论文写作框架

```
1. Introduction
   - RTB 中 win rate prediction 的重要性
   - 现有方法的局限

2. Related Work
   - Sequence modeling for RTB
   - Counterfactual inference
   - Deep learning for ads

3. Methodology
   - DeepWin architecture
   - Training strategy

4. Experiments
   - Dataset analysis (关键：true_value 与 win_label 无关)
   - Baseline comparison
   - Ablation study

5. Conclusion
   - 时序建模是关键
   - Counterfactual inference 需要特定数据条件
```

## 🎯 后续工作

1. **真实数据验证**: 在 Criteo/iPinYou 数据集上测试
2. **在线学习**: 处理市场动态变化
3. **多目标优化**: Win rate + CTR + CVR 联合预测

## 📄 详细文档

- [FINAL_SUMMARY.md](FINAL_SUMMARY.md) - 项目完整总结
- [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) - 实验详细报告
- [results/FINAL_ANALYSIS.md](results/FINAL_ANALYSIS.md) - 深度分析
- [ICLR2026_PAPER_RESULTS.md](ICLR2026_PAPER_RESULTS.md) - ICLR 2026 论文复现

## 👑 作者

秦始皇 - AI Research Assistant

---

*最后更新：2026-04-02*
