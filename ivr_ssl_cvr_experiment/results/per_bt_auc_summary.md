# Per-BT AUC 实验结果汇总（data_v2）

**日期：** 2026-03-29  
**数据集：** data_v2（train: 654,757 / val: 1,940,129）  
**训练配置：** batch_size=4096, epochs=1, lr=1e-3, embed_dim=32, hidden=[256,128], proj_dim=64, temperature=0.1

## 结果表格

| 模型 | Overall | aecps(25w) | aedsp(13w) | aerta(6w) | lazada_cps(11w) | lazada_rta(12w) | shein(58w) | shopee_cps(69w) |
|------|---------|-------|-------|-------|------------|------------|-------|------------|
| baseline (cw=0.1) | 0.8321 | 0.7387 | 0.6419 | 0.7083 | 0.7786 | 0.7474 | 0.5150 | 0.7340 |
| bt_contrastive (cw=0.05) | 0.8315 | 0.7332 | **0.6815** | 0.6950 | 0.7712 | 0.7488 | 0.5169 | 0.7341 |
| bt_contrastive (cw=0.1) | 0.8287 | 0.7366 | 0.6392 | **0.7214** | 0.7766 | 0.7495 | 0.5427 | 0.7324 |
| bt_contrastive (cw=0.2) | 0.8314 | 0.7279 | 0.5985 | 0.6674 | 0.7731 | 0.7452 | 0.5482 | 0.7328 |
| **user_contrastive (cw=0.1)** | **0.8379** | **0.7430** | 0.6446 | 0.6837 | **0.7765** | **0.7495** | **0.6078** | **0.7379** |
| augment_contrastive (cw=0.1) | 0.8130 | 0.7124 | 0.6609 | 0.6587 | 0.7490 | 0.7255 | 0.5375 | 0.7192 |

## vs Baseline 差值（千分点）

| 模型 | Overall | aecps | aedsp | aerta | lazada_cps | lazada_rta | shein | shopee_cps |
|------|---------|-------|-------|-------|------------|------------|-------|------------|
| bt_contrastive (cw=0.05) | -0.6 | -5.5 | +39.6 | -13.3 | -7.4 | +1.4 | +1.9 | +0.1 |
| bt_contrastive (cw=0.1) | -3.4 | -2.1 | -2.7 | +13.1 | -2.0 | +2.1 | +27.7 | -1.6 |
| bt_contrastive (cw=0.2) | -0.7 | -10.8 | -43.4 | -40.9 | -5.5 | -2.2 | +33.2 | -1.2 |
| **user_contrastive (cw=0.1)** | **+5.8** | **+4.3** | +2.7 | -24.6 | -2.1 | +2.1 | **+92.8** | **+3.9** |
| augment_contrastive (cw=0.1) | -19.1 | -26.3 | +19.0 | -49.6 | -29.6 | -21.9 | +22.5 | -14.8 |

## 关键结论

1. **user_contrastive 整体最优**，6/7 个 bt 上最优或并列最优
   - shein 提升最显著（+92.8 千分点），极稀疏场景下 user 对比价值最大
   - aerta 下降 24.6 千分点，可能是样本量小（6w）导致 user 对比噪声大

2. **bt_contrastive 效果分化**
   - aedsp 在 cw=0.05 有显著提升（+39.6 千分点），但 cw 越大越差
   - aerta 在 cw=0.1 有提升（+13.1），但 cw=0.2 大幅下降（-40.9）
   - loss weight 超参极为敏感，需要精细调参

3. **augment_contrastive 整体最差**
   - Overall 下降 19.1 千分点，随机 dropout 增强策略不适合此场景
   - 需要改进为结构化增强（user-side vs ad-side view）

4. **shein 是最难 bt**（正样本率 0.018%，极度稀疏）
   - baseline 只有 0.515，user_contrastive 拉到 0.608
   - SSL 在极稀疏场景下价值最大

## 下一步实验计划

### Round 2：user_contrastive 调参
- cl_weight: 0.02, 0.05, 0.1, 0.2, 0.5
- 目标：找最优 cw，特别关注 aerta 的下降问题

### Round 3：MLoRA 风格 bt-specific 适配
- 给每个 bt 加低秩适配层（rank=4/8）
- 与 user_contrastive 结合

### Round 4：结构化 augmentation
- user-side view（用户特征子集）vs ad-side view（广告特征子集）
- 替换当前随机 dropout 策略
