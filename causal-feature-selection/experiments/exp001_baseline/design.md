# exp001: Baseline WideDeep

## 目标
建立 CTR 预估 baseline，验证跨 business_type 场景下的 OOD 泛化问题确实存在。

## 配置

| 参数 | 值 |
|------|----|
| 模型 | WideDeep |
| embedding_size | 8 |
| dnn_hidden_units | [1024, 512, 256, 128] |
| dropout | 0.3 |
| lr | 5e-5 |
| batch_size | 512 |
| epochs | 1 |
| 优化器 | AdamW |

## 数据划分

| 集合 | business_type | 样本数 |
|------|--------------|--------|
| 训练集 | 0,1,2,3,4,5,6,11 (top 8) | 3,045,806 |
| 测试集 | 7,8,10,13,16 (长尾) | 204,951 |

- 数据集：IVR v16 CTCVR Sample（预编码版本）
- 特征数：125（排除 deviceid）
- 标签：click_label（CTR 任务）

## 假设
- H1: 模型在 OOD BT 上的 AUC 会显著低于 in-domain
- H2: 不同 BT 之间的 PCOC 差异大，说明模型校准性跨域不稳定

## 脚本
`scripts/run_baseline.py`
