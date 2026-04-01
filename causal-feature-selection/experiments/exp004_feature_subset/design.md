# exp004: 特征子集选择实验

**日期**: 2026-04-01  
**状态**: 🔄 进行中

## 目标

基于 Phase 1 特征重要性分析结果，验证：
1. 去掉虚假相关特征后，OOD AUC 是否提升
2. 只保留因果稳定特征后，跨 BT 泛化是否改善
3. 量化虚假相关特征对 OOD 性能的负面影响

## 假设

- **H1**: 去掉虚假相关特征（`os`, `duf_inner_dev_pkg_pur_11_30d`, `duf_inner_dev_pkg_pur_61_180d`）后，OOD AUC 提升
- **H2**: 只保留因果稳定特征（17个）后，PCOC 方差减小（校准更稳定）
- **H3**: 特征子集选择比 IRM 更有效（对比 exp003）

## 实验设计

### 特征子集方案

| 方案 | 特征数 | 说明 |
|------|--------|------|
| A: 全量特征 (baseline) | 125 | exp001 结果，AUC=0.7545 |
| B: 去掉虚假相关特征 | 122 | 去掉 os, duf_inner_dev_pkg_pur_11_30d, duf_inner_dev_pkg_pur_61_180d |
| C: 只保留因果稳定特征 | 17 | exp002 识别的因果候选特征 |
| D: 因果特征 + Top 重要特征 | ~40 | 因果特征 + EmbNorm Top20 |

### 虚假相关特征（来自 exp002）
- `os`: ID rank=90, OOD rank=125（完全失效）
- `duf_inner_dev_pkg_pur_11_30d`: ID rank=50, OOD rank=102（-52 漂移）
- `duf_inner_dev_pkg_pur_61_180d`: ID rank=52, OOD rank=101（-49 漂移）

### 因果稳定特征（来自 exp002，17个）
- `is_interstitial_ad`, `business_type`, `demand_pkgname`, `connectiontype`
- `duf_inner_dev_pkg_imp_31_60d`, `duf_inner_dev_pkg_re_bucket_15d`
- `duf_inner_dev_pkg_imp_bucket_15d`, `offerid`, `duf_inner_dev_pkg_imp_bucket_3d`
- `duf_inner_dev_pkg_imp_bucket_7d`, `duf_inner_dev_pkg_re_bucket_7d`
- `duf_inner_dev_pkg_re_bucket_3d`, `duf_inner_dev_pkg_imp_4_10d`
- `duf_inner_dev_pkg_open_60d`, `huf_deviceid_demand_pkgname_re_24h`
- `duf_inner_dev_pkg_open_4_10d`, `adx`

## 模型配置

- 模型：WideDeepBaseline（同 exp001/003）
- lr=5e-5, batch_size=512, epochs=1
- 每个方案重新初始化（torch.manual_seed(42)）
- 训练集：BT=[0,1,2,3,4,5,6,11]，测试集：BT=[7,8,10,13,16]

## 评估指标

- 整体 AUC + PCOC + LogLoss
- 分 BT 的 AUC + PCOC
- PCOC 跨 BT 方差（稳定性）
