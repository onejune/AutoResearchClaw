# P9 · DSP 分广告主自适应建模 — 设计方案

> 版本：v0.1 | 创建：2026-03-26 | 状态：调研 & 设计阶段

---

## 1. 问题定义

### 1.1 背景

rec-autopilot 实验表明，当前全局单模型（WideDeep）在 ivr_sample_v16 数据上最优 AUC 为 **0.8374**，评估时按 business_type 分组计算。各广告主（shein / aliexpress / shopee / lazada）在以下维度存在显著差异：

| 差异维度 | 说明 |
|---------|------|
| **样本量** | shein 负样本采样 1%，其他采 10%，头尾差距悬殊 |
| **转化路径** | shein 直接购买 vs lazada/shopee 平台跳转，CVR 分布不同 |
| **用户群体** | 不同广告主覆盖的地区、设备、用户行为习惯差异大 |
| **目标类型** | objective_type 不同（SALES_WEBSITE vs APP_INSTALL 等） |

**核心假设**：用一个全局模型统一学习所有广告主，会导致各广告主特有的转化模式被平均化，分广告主建模或自适应建模可以提升各广告主的 AUC。

### 1.2 目标

- **主目标**：提升各 business_type 分组 AUC，尤其是非 shein 的长尾广告主
- **约束**：不能显著增加推理延迟（线上 DSP RTB 场景）
- **评估**：分 business_type 的 AUC / PCOC，与全局单模型 baseline 对比

---

## 2. 方案选型

### 2.1 候选方案对比

| 方案 | 核心思想 | 参数量 | 实现难度 | 适合场景 |
|------|---------|--------|---------|---------|
| **独立模型** | 每个广告主单独训练 | 4x | 低 | 数据充足的头部广告主 |
| **Shared-Bottom** | 共享底层 + 各自 tower | 1.2x | 低 | 弱相关任务 |
| **MMoE** | 多 expert + 门控路由 | 1.5x | 中 | 任务相关性中等 |
| **PLE** | 专属 expert + 共享 expert + CGC | 2x | 中 | 任务差异较大 |
| **STAR** | 共享参数 × 场景参数（element-wise乘法）+ 辅助网络 | 1.8x | 中 | 多场景，参数高效 |
| **AdaSparse** | 场景感知的稀疏特征剪枝 | 1.3x | 中高 | 特征空间大，场景差异在特征上 |
| **AdaptDHM** | 动态聚类路由，不依赖显式 domain label | 1.5x | 高 | domain label 噪声大时 |

### 2.2 方案选择

**推荐优先级：STAR > PLE > MMoE > Shared-Bottom（baseline）**

理由：
1. **STAR** 最契合本场景：business_type 是明确的场景标签，STAR 的"共享参数 × 场景参数"设计参数高效，且有辅助网络兜底，阿里在电商广告场景验证充分
2. **PLE** 作为第二梯队：CGC 结构对各广告主差异大的情况更鲁棒，RecSys 2020 工业落地经验丰富
3. **MMoE** 作为轻量 baseline 对比
4. **Shared-Bottom** 作为最弱 baseline，验证分场景是否有收益

**暂不做**：AdaptDHM（聚类路由复杂，且我们有明确 domain label，不需要无监督发现）、M2M（Transformer 在 CPU 训练环境下太重）

---

## 3. 技术方案详述

### 3.1 整体架构

```
输入特征（精简特征集）
        │
        ▼
  Embedding Layer
  （共享 embedding table）
        │
        ├─── domain_indicator = business_type_id
        │
        ▼
  多场景建模层（STAR / PLE / MMoE）
        │
        ▼
  各广告主 Tower Head
        │
        ▼
  sigmoid → CTR 预估
```

### 3.2 核心模型：STAR

参考 Scenario-Wise-Rec 中的 `star.py` 实现，适配 MetaSpore 训练框架。

**STAR 核心机制**：
- **Domain Normalization**：共享 γ/β + 场景专属 γ/β，对 embedding 做场景感知归一化
- **Star Topology FCN**：共享网络参数 W_shared，场景网络参数 W_domain，前向计算 `W = W_shared ⊙ W_domain`（element-wise 乘法）
- **辅助网络**：全量数据训练的辅助 MLP，输出与主网络相加，防止长尾场景过拟合

```
STAR 参数规模估算（以 input_dim=256, fcn=[512,256,64] 为例）：
- 共享 FCN：256×512 + 512×256 + 256×64 + 64×1 ≈ 0.3M
- 4个场景专属 FCN：4 × 0.3M = 1.2M
- 辅助网络：~0.1M
- 总计：~1.6M（vs 全局模型 ~0.3M，约 5x）
```

### 3.3 数据管道适配

复用 rec-autopilot 的 `BaseTrainFlow` 采样逻辑，**新增 domain_indicator 字段**：

```python
# 在 _read_dataset_by_date 中新增
DOMAIN_MAP = {
    'shein': 0,
    'ae': 1,      # aliexpress 前缀匹配
    'shopee': 2,
    'lazada': 3,
}

df = df.withColumn("domain_indicator", 
    F.when(F.col("business_type") == "shein", 0)
     .when(F.col("business_type").startswith("ae"), 1)
     .when(F.col("business_type").startswith("shopee"), 2)
     .when(F.col("business_type").startswith("lazada"), 3)
     .otherwise(0)  # fallback to shein
)
```

**训练配置**：
- 训练窗口：7 天（vs rec-autopilot 的 3 天）
- 验证：1 天
- 采样策略：沿用 `random_sample()`（shein 负样本 1%，其他 10%）

### 3.4 特征配置

特征由老板提供配置文件，直接使用，不做额外筛选。`business_type` 作为 domain_indicator 的来源，同时也保留在特征中（作为 embedding 特征）。

---

## 4. 实验计划

### 4.1 实验矩阵

| 实验 ID | 模型 | 说明 | 预期 AUC delta |
|--------|------|------|--------------|
| exp_adv_001 | WideDeep（全局） | baseline，7天训练 | 参考值 |
| exp_adv_002 | Shared-Bottom | 最弱多场景 baseline | +0~0.002 |
| exp_adv_003 | MMoE（4 experts） | 轻量多场景 | +0.002~0.005 |
| exp_adv_004 | PLE（1 level, 2 specific + 1 shared） | 中等复杂度 | +0.003~0.008 |
| exp_adv_005 | STAR（fcn=[512,256,64]） | **主力方案** | +0.005~0.010 |
| exp_adv_006 | STAR（fcn=[1024,512,256]） | 更大容量 | 消融 |
| exp_adv_007 | STAR + 独立 embedding | 场景专属 embedding | 消融 |

### 4.2 评估维度

```
主指标：
  - 各 business_type 分组 AUC（shein / ae / shopee / lazada）
  - 整体 AUC（加权平均）
  - PCOC（校准度）

辅助指标：
  - 训练时间（7天数据，CPU 环境）
  - 模型参数量
```

### 4.3 消融实验（视主实验结果决定是否做）

- 共享 embedding vs 场景专属 embedding
- Domain Normalization 的作用（去掉 DN 的 STAR）
- 辅助网络的作用（去掉 aux_net 的 STAR）
- 训练天数：3天 vs 7天 vs 14天

---

## 5. 工程实现计划

### 5.1 目录结构

```
exp_advertiser_adaptive/
├── DESIGN.md               # 本文档
├── conf/
│   ├── base.yaml           # 训练配置（7天窗口，复用 rec-autopilot 路径）
│   ├── combine_schema      # 特征配置（由老板提供）
│   └── experiments/        # 各实验 yaml
│       ├── exp_adv_001_baseline.yaml
│       ├── exp_adv_003_mmoe.yaml
│       ├── exp_adv_004_ple.yaml
│       └── exp_adv_005_star.yaml
├── src/
│   ├── data_loader.py      # 数据管道（复用 BaseTrainFlow 采样，新增 domain_indicator）
│   ├── features.py         # 特征定义（SparseFeature / DenseFeature）
│   ├── models/
│   │   ├── __init__.py
│   │   ├── shared_bottom.py
│   │   ├── mmoe.py         # 参考 Scenario-Wise-Rec
│   │   ├── ple.py          # 参考 Scenario-Wise-Rec
│   │   └── star.py         # 参考 Scenario-Wise-Rec，核心模型
│   ├── trainer.py          # 训练循环（PyTorch，独立于 MetaSpore）
│   └── main.py             # 入口，解析 yaml 配置
├── scripts/
│   ├── run_experiment.sh   # 单实验运行
│   └── run_all.sh          # 批量运行
├── experiments/            # 实验记录（自动生成）
├── leaderboard.json        # 排行榜
└── README.md
```

### 5.2 与 rec-autopilot 的关系

| 复用内容 | 方式 |
|---------|------|
| 数据路径、Spark 读取 | 直接 import `BaseTrainFlow` 或复制采样逻辑 |
| 采样策略（random_sample） | 直接复用 |
| 评估逻辑（分 business_type AUC） | 复用 `_eval_by_df` 逻辑 |
| 特征 combine_schema | 由老板提供新配置 |

**不复用**：MetaSpore 模型结构（用 PyTorch 原生实现多场景模型，更灵活）

### 5.3 关键实现细节

**domain_indicator 处理**：
- 训练时：从 `business_type` 字段映射为整数 domain id，作为额外输入
- 模型内：既作为路由信号（选择哪个 expert/tower），也作为 embedding 特征

**MetaSpore vs PyTorch 选择**：
- MetaSpore 的 PS 架构适合超大规模 embedding，但多场景模型结构灵活性差
- 建议：**embedding 层用 MetaSpore PS**（处理高基数稀疏特征），**多场景网络层用 PyTorch**（灵活实现 STAR/PLE）
- 如果实现复杂度太高，可先用纯 PyTorch + 小 embedding 验证方案可行性，再接入 MetaSpore

---

## 6. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 长尾广告主（lazada/shopee）样本太少，场景专属参数欠拟合 | 中 | 高 | STAR 的辅助网络 + 共享参数乘法天然缓解；可加 domain-aware 正则 |
| 7天数据训练时间过长（CPU 环境） | 中 | 中 | 先用 3天数据验证方案，确认有收益再上 7天 |
| business_type 前缀匹配不准确（ae_xxx 种类多） | 低 | 低 | 统计各 business_type 值分布，确认映射规则 |
| STAR 的 element-wise 乘法导致梯度消失 | 低 | 中 | 参考原论文初始化策略（domain 参数初始化为全 1） |

---

## 7. 参考文献

| 论文 | 会议 | 核心贡献 |
|------|------|---------|
| STAR: One Model to Serve All | CIKM 2021 | Star Topology FCN + Domain Normalization |
| PLE: Progressive Layered Extraction | RecSys 2020 | CGC 结构，专属 + 共享 expert |
| MMoE: Modeling Task Relationships | KDD 2018 | 多 expert + 门控，多任务经典 |
| AdaSparse | CIKM 2022 | 场景感知特征稀疏化 |
| Scenario-Wise-Rec Benchmark | - | 多场景统一 benchmark，含上述模型实现 |

---

## 8. 下一步行动

- [ ] 老板提供特征配置文件（combine_schema）
- [ ] 统计 ivr_sample_v16 各 business_type 的样本量和 CVR 分布
- [ ] 搭建 exp_advertiser_adaptive 目录骨架
- [ ] 实现 data_loader.py（复用采样逻辑 + 新增 domain_indicator）
- [ ] 实现 STAR 模型（参考 Scenario-Wise-Rec/star.py，适配本项目特征格式）
- [ ] 跑 exp_adv_001（7天 baseline），确认流程 OK
- [ ] 跑 exp_adv_005（STAR），对比 baseline
