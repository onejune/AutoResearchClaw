# DSP 分广告主自适应建模 — 实验报告

> 项目：P9 · Advertiser Adaptive | 最后更新：2026-03-27

---

## 一、数据集

### 数据来源
- **数据集**：ivr_sample_v16（内部 DSP IVR 广告数据）
- **原始路径**：`/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet/YYYY-MM-DD/`

### 实验数据窗口
| 集合 | 日期范围 | 行数 | 正样本数 | 正样本率 |
|------|---------|------|---------|---------|
| 训练集 | 2026-03-01 ~ 2026-03-14（14天） | 22,412,695 | 1,826,480 | 8.15% |
| 验证集 | 2026-03-15 ~ 2026-03-17（3天） | 4,786,249 | 371,341 | 7.76% |

### Domain 分布（训练集）
| 广告主 | domain id | 训练集行数 | 占比 |
|--------|-----------|-----------|------|
| shopee | 2 | ~949万 | 42% |
| aliexpress | 1 | ~792万 | 35% |
| lazada | 3 | ~440万 | 20% |
| shein | 0 | ~60万 | 3% |

### 采样策略
- 正样本：全部保留
- 负样本：shein 保留 1%，其他广告主保留 10%（seed=42）
- shein 负样本采样率低（1%）的原因：shein 原始负样本量远大于其他广告主，1% 采样使各 domain 样本量更均衡

### 特征配置
- 特征文件：`combine_schema`（186 个稀疏特征）
- 特征处理：字符串 hash → embedding index，vocab_size=100,000，缺失值填 index 0

---

## 二、模型配置

### 公共配置（base.yaml）
| 参数 | 值 |
|------|----|
| embed_dim | 8 |
| vocab_size | 100,000 |
| batch_size | 1,024 |
| learning_rate | 5e-5 |
| weight_decay | 1e-5 |
| epochs | 3 |
| early_stop_patience | 3 |
| device | cpu |

### 各模型结构
| 模型 | 核心结构 | 参数量（估算） |
|------|---------|--------------|
| Baseline | DNN [1024,512,256,128], dropout=0.3 | ~151M |
| Shared-Bottom | bottom [512,256] + tower [128,64] | ~1.4M |
| MMoE | 4 experts, gate per domain | ~1.8M |
| PLE | 2 specific + 1 shared expert, CGC | ~2.9M |
| STAR | shared FCN ⊙ domain FCN + aux net | ~3.6M |

---

## 三、实验结果

### exp_001_baseline（分层采样修复后，2026-03-26 23:03）

> ⚠️ 注意：2026-03-26 18:47 发现分层采样 bug（小 domain 样本不足），已修复后重跑。
> 本结果为修复后的正确结果。

| Domain | AUC | PCOC | LogLoss | 正样本数 | 总样本数 |
|--------|-----|------|---------|---------|---------|
| **Overall** | **0.7815** | 1.181 | 0.2370 | 37,338 | 478,623 |
| aliexpress | 0.7983 | 0.971 | 0.1106 | 4,027 | 142,880 |
| lazada | 0.7621 | 0.924 | 0.3279 | 12,013 | 95,795 |
| shopee | 0.7293 | 1.375 | 0.2866 | 20,882 | 224,518 |
| shein | 0.6632 | 0.895 | 0.1203 | 416 | 15,430 |

耗时：39.4 min

### exp_002~005（分层采样修复后重跑，进行中）

> 状态：2026-03-27 07:01 启动，预计 2026-03-27 10:00 完成，结果自动更新。

---

## 四、Bug 记录

### Bug-001：分层采样导致小 domain 样本不足（2026-03-26 18:47 发现并修复）

**问题**：`DEBUG_SAMPLE` 模式下，原代码用 `Subset(ds, range(int(len(ds)*ratio)))` 截取前 N%，
导致 shein（占比仅 3%）在 1% 采样后只有 42 个正样本，低于评估过滤阈值，metrics.json 中 shein 缺失。

**影响**：exp_002~005（2026-03-26 15:09~17:48 的结果）中 shein 数据不可信，aliexpress 等 domain 的 AUC 也因采样偏差失真（如 aliexpress AUC 从正确的 0.798 降至 0.706）。

**修复**：改为按 domain 分层采样，每个 domain 至少保留 50 个样本：
```python
def stratified_sample(dataset, ratio):
    for d in np.unique(domains):
        d_indices = np.where(domains == d)[0]
        n_sample = max(int(len(d_indices) * ratio), 50)
        sampled = np.random.choice(d_indices, n_sample, replace=False)
        indices.extend(sampled.tolist())
```

**重跑**：exp_001 已用修复后代码重跑（结果见上），exp_002~005 于 2026-03-27 07:01 重启。

---

## 五、关键发现（持续更新）

1. **全局 Baseline AUC = 0.7815**，作为多场景模型的对比基准
2. **各 domain AUC 差异显著**：aliexpress 最高（0.798），shein 最低（0.663）
3. **shein 样本量极少**（验证集仅 416 个正样本，占 1.1%），模型对 shein 的学习效果受限
4. **shopee PCOC 偏高**（1.375），预测值偏高，需关注校准问题

---

## 六、下一步计划

- [ ] 等待 exp_002~005 重跑结果（预计 2026-03-27 上午完成）
- [ ] 对比 5 个模型，确认多场景建模是否有收益
- [ ] 若 STAR/PLE 有收益，启动消融实验（exp_006~007）
- [ ] 考虑 shein 单独优化（样本量太少，可能需要 Domain Adaptation）
