# exp002: 特征重要性分析

## 目标
对比三种特征重要性方法在 in-domain vs OOD 场景下的排名差异，识别"因果特征"和"虚假相关特征"。

## 方法

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **Embedding Norm (EN)** | 训练后 embedding 权重 L2 范数 | 零额外计算 | 受词频影响 |
| **Gradient Importance (GI)** | loss 对 embedding 梯度的 L2 范数均值 | 计算快，一次反向传播 | 局部近似 |
| **Permutation Importance (PI)** | 打乱特征后 AUC 下降幅度 | 最可靠，模型无关 | 计算慢 |

## 数据

| 集合 | 说明 | 样本数 |
|------|------|--------|
| In-domain | 训练 BT 中随机采样 | 50,000 |
| OOD | 全量测试集 (BT 7,8,10,13,16) | 204,951 |

## 假设
- H1: 因果特征（如广告类型、网络类型）在 in-domain 和 OOD 下排名稳定
- H2: 用户行为统计特征（如购买历史）在 OOD 下排名下降（虚假相关）
- H3: 三种方法的 top-20 排名有较高 Spearman 相关性

## 脚本
`scripts/run_feature_importance.py`
