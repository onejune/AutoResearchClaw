# MetaCVR：广告 Campaign 冷启动 Purchase 预估

> **项目路径**：`/mnt/workspace/open_research/autoresearch/exp_meta_coldstart/`
> **关联项目**：`../exp_multitask/`（数据复用）
> **目标会议**：KDD / WWW / RecSys

完整实验分析见 → [`experiment_report.md`](./experiment_report.md)（实验开始后创建）

---

## 背景与动机

### 业务场景

在 DSP 广告系统（IVR）中，广告主每天都会新建 Campaign Set（投放计划），用于推广新商品或开启新的营销活动。新建的 Campaign Set 在上线初期**几乎没有历史曝光和转化数据**，而传统的 Purchase 预估模型（如 MMoE、PLE）严重依赖行为特征的统计积累，导致：

- 新 Campaign Set 的 Purchase 预估准确率极低
- 广告系统倾向于保守出价，新广告获量困难
- 广告主冷启动期 ROI 差，容易流失

这是工业界广告系统中**普遍存在且影响显著**的问题。

### 为什么选 Campaign Set 粒度？

| 粒度 | 总数 | 数据充足比例 | 新建频率 | 冷启动真实性 | 预估复杂度 |
|------|------|--------------|----------|--------------|------------|
| Ad | ~千万 | < 1% | 极高 | ✅ 强 | 低 |
| Creative | ~百万 | < 5% | 高 | ✅ 强 | 低 |
| Campaign | ~十万 | ~20% | 中 | ✅ 中 | 中 |
| **Campaign Set** | **~万** | **~50%** | **低** | **✅ 弱** | **高** |

我们选择 Campaign Set 粒度是因为：
1. **数据稀疏性适中**：既有冷启动需求，又有一定数据支撑
2. **业务粒度合理**：对应广告主投放策略，预估结果可直接指导出价
3. **预估复杂度高**：涉及预算、人群、创意组合等多重因素，需要更复杂的模型

---

## 问题定义

### 形式化描述

给定一个新 Campaign Set $C_{new}$，仅有 $K$ 个样本 $\mathcal{D}_{support} = \{(x_i, y_i)\}_{i=1}^{K}$，其中 $x_i$ 为特征向量（包括 Campaign Set 级别特征、用户特征、上下文特征等），$y_i \in \{0,1\}$ 为 Purchase 标签。

目标是快速构建一个模型 $f_{\theta}$，使其在该 Campaign Set 的未来样本上有良好的 Purchase 预估效果：

$$\max_f \mathbb{E}_{(x,y) \sim \mathcal{D}_{new}} [\text{AUC}(f(x), y)]$$

### 技术挑战

1. **数据极度稀疏**：K << N，传统监督学习方法失效
2. **领域迁移**：新 Campaign Set 与历史数据可能存在分布偏移
3. **实时性要求**：广告系统要求秒级响应，不允许长时间训练

---

## 解决方案

### 1. Meta-Learning（元学习）

将每个 Campaign Set 视为一个独立的任务，通过在大量历史任务上训练元模型，使其具备快速适应新任务的能力。

#### MAML（Model-Agnostic Meta-Learning）

$$\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{D}_{support}}(f_\theta)$$
$$\theta \leftarrow \theta - \beta \nabla_\theta \mathcal{L}_{\mathcal{D}_{query}}(f_{\theta'})$$

#### FOMAML（First-Order MAML）

忽略二阶梯度，只保留一阶梯度项，提高训练效率。

#### ANIL（Almost No Inner Loop）

只对模型的最后几层进行 inner loop 更新，大幅减少计算量。

#### ProtoNet（原型网络）

为每个类别学习一个原型表示，通过距离度量进行分类。

### 2. Adapter Tuning（适配器调优）

在预训练模型中插入小型适配器层，只训练少量参数实现快速适配。

$$h = W_{down} \cdot x$$
$$h = \sigma(h)$$
$$h = W_{up} \cdot h$$
$$x_{out} = x + LayerNorm(h)$$

---

## 实验设计

### 数据集构建

- **数据来源**：IVR 广告系统日志
- **时间范围**：2023年某月数据
- **样本量**：~86万条样本，265个 Campaign Set
- **特征维度**：124维稀疏特征
- **正样本率**：~7.5%

### 任务划分

- **Meta-train**：212个 Campaign Set
- **Meta-test**：53个 Campaign Set
- **K-shot**：支持集大小，分别测试 K=10, 50, 100, 500

### 基线方法

1. **Global Model**：全量数据训练单一模型
2. **Pretrain + Fine-tune**：预训练后用K个样本微调
3. **Per-Task Model**：每个任务单独训练（上界）

---

## 项目结构

```
exp_meta_coldstart/
├── README.md                 # 项目说明
├── experiment_report.md      # 实验报告（持续更新）
├── data/                     # 数据目录
│   ├── train.pkl             # 训练数据
│   └── meta_info.pkl         # 元数据信息
├── src/                      # 核心代码
│   ├── config.py             # 配置文件
│   ├── data.py               # 数据处理
│   ├── models.py             # 模型定义
│   ├── baseline.py           # 基线方法
│   ├── meta_learner.py       # 元学习方法
│   └── adapter.py            # 适配器方法
├── scripts/                  # 实验脚本
│   ├── run_exp_001_baseline.py    # 基线实验
│   ├── run_exp_002_maml.py        # MAML实验
│   ├── run_exp_003_kshot.py       # K-shot分析
│   ├── run_exp_004_fixed_maml.py  # 修复版MAML
│   ├── run_exp_005_fixed_baseline.py # 修复版基线
│   └── run_exp_006_adapter.py     # Adapter实验
└── results/                  # 实验结果
    ├── exp_001_baseline.md
    ├── exp_002_maml.md
    ├── exp_003_kshot.md
    ├── exp_004_fixed_maml.md
    ├── exp_005_fixed_baseline.md
    └── exp_006_adapter.md
```