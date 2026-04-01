# exp003: IRM 多环境训练

## 目标
用 Invariant Risk Minimization (IRM) 训练模型，学习跨 business_type 稳定的不变特征表示，
改善 OOD 场景下的 AUC 和 PCOC。

## 背景
IRM (Arjovsky et al., 2019) 的核心思想：
- 将训练数据划分为多个"环境"（这里用 business_type 分组）
- 在每个环境上同时最小化 ERM loss 和"不变性惩罚"
- 不变性惩罚：要求每个环境上的最优分类器是同一个线性分类器
- 数学形式：min_{Φ} Σ_e R^e(Φ) + λ · Σ_e ||∇_{w|w=1} R^e(w·Φ)||²

## 环境划分方案

| 环境 | business_type | 样本数（约） | 说明 |
|------|--------------|------------|------|
| env_0 | 0, 1 | ~1.37M | 最大两个 BT |
| env_1 | 2, 3, 4 | ~0.76M | 中等规模 |
| env_2 | 5, 6, 11 | ~0.91M | 剩余训练 BT |
| test  | 7, 8, 10, 13, 16 | ~0.20M | OOD 测试 |

## 配置

| 参数 | 值 |
|------|----|
| 模型 | WideDeep（同 exp001） |
| IRM λ (penalty_weight) | 1.0, 10.0, 100.0（消融） |
| lr | 5e-5 |
| batch_size | 512（每个环境独立采样） |
| epochs | 1 |
| 特征 | 全量 125 特征 |

## 假设
- H1: IRM 训练后整体 OOD AUC 优于 baseline (exp001: 0.7545)
- H2: PCOC 跨 BT 方差缩小（校准性更稳定）
- H3: λ 越大，OOD 提升越明显，但 in-domain 可能略有下降

## 对比
| 实验 | 方法 | 预期 |
|------|------|------|
| exp001 | ERM baseline | AUC=0.7545, PCOC=0.7297 |
| exp003 | IRM (λ=1) | AUC↑?, PCOC 方差↓? |
| exp003 | IRM (λ=10) | - |
| exp003 | IRM (λ=100) | - |

## 脚本
`scripts/run_exp003_irm.py`
