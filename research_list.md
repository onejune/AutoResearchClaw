# Research List

> 所有研究方向的状态追踪与核心内容记录。及时更新。
> 最后更新：2026-03-25

---

## 研究方向总览

| # | 方向 | 状态 | 数据集 | 目标会议 | 代码路径 |
|---|------|------|--------|---------|---------|
| P0 | 多任务 CVR 建模 | ⏳ 等数据 | Ali-CCP | KDD / WWW | `exp_multitask/` |
| P1 | 连续特征自动离散化 | 🔬 实验中 | Criteo Standard | KDD / WWW / SIGIR | `exp_continuous_features/` |
| P2 | Meta-Learning 冷启动 CVR | 📋 待启动 | Ali-CCP | KDD / WWW | - |
| P3 | 稀疏转化信号自监督增强 | 📋 待排期 | Ali-CCP | KDD / WWW | - |
| P4 | 稀疏反馈 + 知识蒸馏 | 📋 待排期 | Ali-CCP | KDD / SIGIR | - |
| P5 | 个性化频次上限预测 | 📋 待排期 | Ali-CCP | KDD / RecSys | - |
| P6 | 频次控制 + 多目标优化 | 📋 待排期 | Ali-CCP | KDD / WWW | - |
| P7 | 延迟奖励 CVR 建模（RL） | 📋 待排期 | Ali-CCP / Criteo | KDD / AAAI | - |
| P8 | 频次控制 RL 策略 | 📋 待排期 | Ali-CCP | KDD / WWW | - |
| ~~P-1~~ | ~~CVR 延迟反馈（FDAM）~~ | ~~⛔ 暂停~~ | ~~Criteo KDD 2014~~ | ~~KDD 2026~~ | ~~-~~ |

**状态说明：** 🔬 实验中 / ⏳ 等数据/条件 / 📋 待启动 / ✅ 完成 / ⛔ 暂停

---

## 详细说明

### P0 · 多任务 CVR 建模

| 项目 | 内容 |
|------|------|
| **核心问题** | 广告场景下 CTR 和 CVR 联合建模，利用多任务学习提升 CVR 预估效果 |
| **对比方法** | Shared-Bottom / ESMM / MMoE / ESCM2 |
| **创新方向** | 待定（需先跑 baseline，分析差异后确定） |
| **数据集** | Ali-CCP（天池 408，~8400万样本，CTR+CVR 双标签） |
| **当前状态** | PyTorch 框架已完成，合成数据验证通过；等 Ali-CCP 数据 |
| **框架路径** | `exp_multitask/`（config.py / data.py / models.py / trainer.py / main.py） |
| **合成数据结果** | Shared-Bottom CTR 0.6352 / ESMM CTR 0.6417 / MMoE CTR 0.6334 / ESCM2 CTR 0.6381 |
| **阻塞原因** | Ali-CCP 数据未下载 |

---

### P1 · 连续特征自动离散化与表示学习

| 项目 | 内容 |
|------|------|
| **核心问题** | 广告/推荐模型中连续数值特征（价格、CTR、年龄等）的最优处理方式 |
| **对比方法** | NoneEncoder（ablation）/ ScalarEncoder（baseline）/ BucketEncoder / AutoDis / NumericEmbedding / FTTransformer / PeriodicEncoder |
| **数据集** | Criteo Standard Display Advertising（4125万行，13个连续特征 + 26个类别特征） |
| **Backbone** | DeepFM（FM + MLP(256,128,64)） |
| **框架路径** | `exp_continuous_features/`（支持多数据集，config 切换） |
| **当前状态** | 100万采样实验完成；**全量实验并行运行中**（4个子进程） |

**已完成实验：**

| 实验 | 数据集 | 样本量 | 最优方法 | 最优AUC | 连续特征贡献 |
|------|--------|--------|---------|---------|------------|
| exp_001 | Criteo Conversion Logs（8特征） | 100万 | NumericEmbedding | 0.8572 | +3.4千分点 |
| exp_002 | Criteo Standard（13特征） | 100万 | NumericEmbedding | 0.7792 | +33.7千分点 |
| exp_003 | Criteo Standard（13特征） | **全量4125万** | 🔄 运行中 | - | - |

**核心发现（截至 exp_002）：**
- NumericEmbedding 在两个数据集上均排名第一，性价比最好
- FTTransformer 效果第二但训练时间是其他方法 2 倍，不划算
- AutoDis 在 100 万样本下未能超越 BucketEncoder，需全量数据验证
- 13个连续特征的贡献（+33.7千分点）远大于8个（+3.4千分点）

---

### P2 · Meta-Learning 冷启动 CVR

| 项目 | 内容 |
|------|------|
| **核心问题** | 新广告缺乏历史数据，用 MAML 学习对冷启动友好的参数初始化 |
| **方法** | MAML + ESMM/ESCM2 结构，少量样本快速适配新广告 CTR+CVR |
| **数据集** | Ali-CCP |
| **当前状态** | 待启动（等 P0 多任务框架跑完后启动） |

---

### P3 · 稀疏转化信号自监督增强

| 项目 | 内容 |
|------|------|
| **核心问题** | 转化事件极稀疏（CVR < 1%），用点击行为构造伪转化标签 |
| **方法** | 对比学习 + 弱监督，端到端增强稀疏 CVR 信号表示 |
| **数据集** | Ali-CCP |
| **当前状态** | 待排期 |

---

### P4 · 稀疏反馈 + 知识蒸馏

| 项目 | 内容 |
|------|------|
| **核心问题** | 用样本充足的 CTR 模型蒸馏知识给稀疏 CVR 模型 |
| **方法** | 单向知识蒸馏（CTR teacher → CVR student），比 ESMM 更灵活 |
| **数据集** | Ali-CCP |
| **当前状态** | 待排期 |

---

### P5 · 个性化频次上限预测

| 项目 | 内容 |
|------|------|
| **核心问题** | 现有固定 frequency cap 忽略用户个体差异 |
| **方法** | 用用户行为序列学习个性化疲劳阈值，动态预测最优展示频次 |
| **数据集** | Ali-CCP |
| **当前状态** | 待排期 |

---

### P6 · 频次控制 + 多目标优化

| 项目 | 内容 |
|------|------|
| **核心问题** | 同时优化 CTR、CVR、用户体验（频次惩罚） |
| **方法** | 多目标帕累托优化 + 频次约束 |
| **数据集** | Ali-CCP |
| **当前状态** | 待排期 |

---

### P7 · 延迟奖励 CVR 建模（RL）

| 项目 | 内容 |
|------|------|
| **核心问题** | 转化延迟建模成 RL 问题：点击即时奖励 + 转化延迟奖励 |
| **方法** | Reward shaping 处理稀疏延迟奖励 |
| **数据集** | Ali-CCP / Criteo KDD 2014 |
| **当前状态** | 待排期 |

---

### P8 · 频次控制 RL 策略

| 项目 | 内容 |
|------|------|
| **核心问题** | 频次分配建模成序列决策（展示/不展示） |
| **方法** | State: 用户历史+当前频次；Action: 展示/不展示；Reward: 转化收益-疲劳惩罚 |
| **数据集** | Ali-CCP |
| **当前状态** | 待排期 |

---

### ~~P-1 · CVR 延迟反馈（FDAM）~~ ⛔ 暂停

| 项目 | 内容 |
|------|------|
| **暂停原因** | 全量数据下 FDAM 持续落后 ES-DFM（0.7289 vs 0.7402），epoch 调参无效 |
| **最佳结果** | ep=8，AUC=0.7304，仍低于 ES-DFM 1.14 千分点 |
| **数据集** | Criteo KDD 2014（1589万行） |
| **代码** | `/mnt/workspace/git_project/AutoResearchClaw/cvr-delayed-feedback-kdd2026/` |
