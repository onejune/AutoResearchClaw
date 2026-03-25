# exp_003：多任务 CVR 建模（Taobao，200万样本，+统计特征）

## 实验信息

| 项目 | 内容 |
|------|------|
| 日期 | 2026-03-25 |
| 数据集 | Taobao UserBehavior |
| 样本量 | 200万（train=160万，val=20万，test=20万） |
| 正样本率 | CTR≈9.6%，CVR≈3.9% |
| 稀疏特征 | user_id, item_id, category_id |
| 数值特征 | 12个（hour, dayofweek + 10个统计特征） |
| Epochs | 1 |
| Batch size | 4096 |
| Embedding dim | 16 |
| MLP dims | (256, 128, 64) |
| 设备 | CPU |

## 新增统计特征（10个）

| 特征 | 说明 |
|------|------|
| u_pv_cnt | 用户总 pv 次数（log1p + z-score） |
| u_buy_cnt | 用户总 buy 次数（log1p + z-score） |
| u_cart_cnt | 用户总 cart 次数（log1p + z-score） |
| u_buy_rate | 用户 buy/pv 比率 |
| u_cart_rate | 用户 cart/pv 比率 |
| i_pv_cnt | 物品总 pv 次数（log1p + z-score） |
| i_buy_cnt | 物品总 buy 次数（log1p + z-score） |
| i_buy_rate | 物品 buy/pv 比率 |
| i_cart_rate | 物品 cart/pv 比率 |
| c_buy_rate | 类目 buy/pv 比率 |

## 实验结果

| 模型 | CTR AUC | CVR AUC | CTCVR AUC | 训练时间 |
|------|---------|---------|-----------|---------|
| Shared-Bottom | 0.7602 | 0.8528 | 0.8685 | 90.6s |
| **ESMM** | 0.7610 | **0.8969** | 0.8800 | 92.3s |
| **MMoE** | **0.7660** | 0.8630 | 0.8722 | 94.5s |
| ESCM2 | 0.7595 | 0.8805 | 0.8776 | 92.8s |

## 与 exp_002（baseline）对比

| 指标 | exp_002（无统计特征） | exp_003（+统计特征） | 提升 |
|------|---------------------|---------------------|------|
| CTR AUC（最佳） | 0.5396 | 0.7660 | **+22.6千分点** |
| CVR AUC（最佳） | 0.5963 | 0.8969 | **+30.1千分点** |
| CTCVR AUC（最佳） | 0.5972 | 0.8800 | **+28.3千分点** |

## 结论

- 统计特征带来巨大提升（+0.20~+0.30 AUC），是最重要的特征工程方向
- **ESMM** CVR AUC 最高（0.8969），符合 ESMM 设计初衷（全空间 CVR 建模）
- **MMoE** CTR AUC 最高（0.7660），多专家结构对 CTR 任务更有效
- 训练时间基本相同（90-95s），统计特征不增加训练开销

## 下一步

- [ ] exp_004：增加更多用户特征（近期行为、类目偏好、序列特征等）
- [ ] Ali-CCP 分层采样实验（并行进行中）
