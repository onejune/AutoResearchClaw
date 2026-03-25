# exp_002：多任务 CVR 建模基线（Taobao，200万样本）

## 实验信息

| 项目 | 内容 |
|------|------|
| 日期 | 2026-03-25 |
| 数据集 | Taobao UserBehavior |
| 样本量 | 200万（train=160万，val=20万，test=20万） |
| 正样本率 | CTR=9.64%，CVR=3.96%，CTCVR=3.96% |
| 特征 | user_id, item_id, category_id（稀疏）+ hour, dayofweek（数值） |
| Epochs | 1 |
| Batch size | 4096 |
| Embedding dim | 16 |
| MLP dims | (256, 128, 64) |
| 设备 | CPU |

## 实验结果

| 模型 | CTR AUC | CVR AUC | CTCVR AUC | 训练时间 |
|------|---------|---------|-----------|---------|
| Shared-Bottom | 0.5391 | 0.5882 | 0.5899 | 85.3s |
| ESMM | 0.5358 | 0.5868 | 0.5916 | 88.2s |
| **MMoE** | **0.5396** | **0.5963** | **0.5972** | 90.6s |
| ESCM2 | 0.5398 | 0.5866 | 0.5912 | 87.6s |

## 分析

- MMoE 在 CVR/CTCVR 上略优，但差距很小
- AUC 在 0.54-0.60，未达到正常水平（>0.65）
- **根本原因：特征太弱**，仅有 ID 特征 + 时间特征，缺乏用户行为统计特征
- 相比 Ali-CCP 基线（全部 ~0.5），有明显提升，说明数据集选择正确

## Bug 修复

- `nn.Embedding(vocab_size)` → `nn.Embedding(vocab_size + 1)`（索引从1开始，需要多一个槽位）

## 下一步

- [ ] exp_003：增加用户/物品统计特征（历史 pv/buy/cart 次数、CTR、购买率等）
- [ ] 对比 exp_002 vs exp_003，量化特征工程的收益
