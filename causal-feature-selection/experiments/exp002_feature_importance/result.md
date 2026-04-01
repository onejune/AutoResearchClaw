# exp002: 特征重要性分析 — 实验结果

**日期**: 2026-04-01  
**状态**: ✅ 完成

## 方法说明

- **EN (Embedding Norm)**: 训练后 embedding 权重 L2 范数
- **GI (Gradient Importance)**: loss 对 embedding 梯度范数，100 batches
- 本次未跑 PI（Permutation Importance），计算量大，后续补充

## 因果特征候选 (in-domain 和 OOD 梯度重要性 top-20 均出现)

| 特征 | ID排名 | OOD排名 | 语义 |
|------|--------|---------|------|
| `is_interstitial_ad` | 7 | **1** | 插屏广告标志 |
| `business_type` | 2 | 2 | 业务类型 |
| `demand_pkgname` | 9 | 3 | 广告主包名 |
| `duf_inner_dev_pkg_imp_31_60d` | 4 | 4 | 用户内网31-60天曝光 |
| `connectiontype` | 10 | 5 | 网络连接类型 |
| `duf_inner_dev_pkg_re_bucket_15d` | 5 | 6 | 用户内网15天收入分桶 |
| `duf_inner_dev_pkg_imp_bucket_15d` | 1 | 7 | 用户内网15天曝光分桶 |
| `offerid` | 13 | 9 | offer ID |
| `duf_inner_dev_pkg_imp_bucket_3d` | 6 | 10 | 用户内网3天曝光分桶 |
| `duf_inner_dev_pkg_imp_bucket_7d` | 8 | 11 | 用户内网7天曝光分桶 |
| `duf_inner_dev_pkg_re_bucket_7d` | 12 | 12 | 用户内网7天收入分桶 |

**规律**: 曝光/收入的"分桶"版本（bucket）比原始值更稳定，说明离散化有助于跨域泛化。

## 排名漂移最大的特征 (虚假相关候选)

| 特征 | ID排名 | OOD排名 | 漂移 | 分析 |
|------|--------|---------|------|------|
| `os` | 90 | **125** | -35 | OOD 时完全失效，OS 分布跨 BT 差异大 |
| `duf_inner_dev_pkg_pur_11_30d` | 50 | 102 | -52 | 购买行为跨域泛化差 |
| `duf_inner_dev_pkg_pur_61_180d` | 52 | 101 | -49 | 同上 |
| `osv` | 113 | 59 | +54 | OOD 时反而更重要（domain-specific 信号？） |
| `language` | 111 | 62 | +49 | 同上 |

## 三种方法 top-20 对比

| 方法 | Top-3 特征 |
|------|-----------|
| GI (in-domain) | imp_bucket_15d, business_type, adx |
| GI (OOD) | is_interstitial_ad, business_type, demand_pkgname |
| Embedding Norm | open_90d, is_reward_ad, outer_open_61_180d |

> EN 与 GI 排名差异较大，说明 embedding 范数受词频影响，不如梯度方法可靠。

## 结论

1. **H1 验证**: 广告属性类特征（is_interstitial_ad, connectiontype, demand_pkgname）跨域稳定 ✅
2. **H2 部分验证**: 购买行为（pur_11_30d, pur_61_180d）OOD 排名大幅下降 ✅；但曝光分桶特征（imp_bucket）意外稳定
3. **H3 不成立**: EN 与 GI 排名相关性低，后续以 GI 为主要方法
4. **关键发现**: **分桶（bucket）特征比原始计数特征更具跨域稳定性**，这是一个值得深入的方向

## 下一步
- exp003: IRM 多环境训练，用 causal_candidates 子集 vs 全特征对比 AUC/PCOC
- 补充 Permutation Importance 验证 causal_candidates 的可靠性

## 输出文件
`results/phase1_feature_importance/`
- `embedding_norm_importance.csv`
- `gradient_importance_indomain.csv`
- `gradient_importance_ood.csv`
- `importance_shift_indomain_vs_ood.csv`
- `summary.json`
