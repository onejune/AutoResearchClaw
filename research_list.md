# Research List

> 所有研究方向的状态追踪与核心内容记录。及时更新。
> 最后更新：2026-03-27

---

## 研究方向总览

| # | 方向 | 状态 | 数据集 | 目标会议 | 代码路径 |
|---|------|------|--------|---------|---------|
| P0 | 多任务 CVR 建模 | ⏳ 等数据 | Ali-CCP | KDD / WWW | `exp_multitask/` |
| P1 | 连续特征自动离散化 | 🔬 实验中 | Criteo Standard | KDD / WWW / SIGIR | `exp_continuous_features/` |
| P2 | Meta-Learning 冷启动 CVR | 🔬 实验中 | IVR（ivr_sample_v16） | KDD / WWW / RecSys | `exp_meta_coldstart/` |
| P3 | 稀疏转化信号自监督增强 | 📋 待排期 | Ali-CCP | KDD / WWW | - |
| P4 | 稀疏反馈 + 知识蒸馏 | 📋 待排期 | Ali-CCP | KDD / SIGIR | - |
| P5 | 个性化频次上限预测 | 📋 待排期 | Ali-CCP | KDD / RecSys | - |
| P6 | 频次控制 + 多目标优化 | 📋 待排期 | Ali-CCP | KDD / WWW | - |
| P7 | 延迟奖励 CVR 建模（RL） | 📋 待排期 | Ali-CCP / Criteo | KDD / AAAI | - |
| P8 | 频次控制 RL 策略 | 📋 待排期 | Ali-CCP | KDD / WWW | - |
| P9 | DSP 分广告主自适应建模 | 🔬 调研中 | ivr_sample_v16（内部） | KDD / WWW / RecSys | - |
| P10 | IVR 自监督对比学习 | 🔬 实验中 | ivr_sample_v16_ctcvr | KDD / WWW | `ivr_ssl_cvr/` |
| P10 | 生成式推荐（LLM × 推荐系统） | 📋 待启动 | Amazon / IVR | RecSys / KDD | - |
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

### P9 · DSP 分广告主自适应建模

| 项目 | 内容 |
|------|------|
| **核心问题** | DSP 电商场景下不同广告主（shein / aliexpress / shopee / lazada）转化行为差异显著，单一全局模型无法充分捕捉各广告主特性 |
| **核心思想** | 以 business_type 为场景/任务粒度，用多任务或多场景建模思想分开学习，同时保留跨广告主的共享知识 |
| **对比方法** | 全局单模型（baseline）/ 各广告主独立模型 / Shared-Bottom / MMoE / PLE / STAR / M2M |
| **创新方向** | 待调研后确定（候选：广告主自适应 adapter、动态专家路由、广告主 embedding 注入） |
| **数据集** | ivr_sample_v16（内部 DSP IVR 数据，7天训练 + 1天验证，正样本全保留，负样本适当采样） |
| **特征策略** | 在 rec-autopilot 现有特征基础上精简（删 IPUA 等低价值特征），聚焦高效特征子集 |
| **评估指标** | 分 business_type 的 AUC / PCOC / LogLoss，重点关注各广告主分别提升情况 |
| **框架** | MetaSpore（复用 rec-autopilot 基础设施） |
| **代码路径** | `exp_advertiser_adaptive/`（DESIGN.md 已创建） |
| **当前状态** | 调研中（文献调研 + 技术路线设计） |
| **目标会议** | KDD / WWW / RecSys |

---

### P10 · 生成式推荐（LLM × 推荐系统）

| 项目 | 内容 |
|------|------|
| **核心问题** | 将推荐问题转化为语言模型生成问题，探索 LLM 在广告/推荐场景的应用潜力 |
| **参考项目** | [MiniMind](https://github.com/jingyaogong/minimind)（64M LLM 从零训练）/ [MiniOneRec](https://github.com/AkaliKong/MiniOneRec)（OneRec 开源复现） |
| **MiniOneRec 核心思路** | SID 构建（RQ-VAE 把商品压缩为离散 token）→ SFT（用户历史序列预测下一商品）→ GRPO 推荐导向 RL |
| **当前状态** | 待启动（先完成 P0/P1 后推进） |
| **目标会议** | RecSys / KDD / WWW |

**三个子方向：**

#### P10-A · 轻量生成式推荐
| 项目 | 内容 |
|------|------|
| **问题** | MiniOneRec 依赖 Qwen2.5 等大模型，计算成本高，工业部署困难 |
| **方案** | 用 MiniMind（64M）替换骨干，研究极小参数量下生成式推荐的效果下限 |
| **核心问题** | 多小的 LLM 能保持生成式推荐的有效性？参数量 vs 推荐质量的 Pareto 边界在哪里？ |
| **数据集** | Amazon（复现 MiniOneRec baseline），再迁移到 IVR |
| **对比基线** | MiniOneRec（Qwen2.5）/ 传统 MMoE |

#### P10-B · 多目标生成式推荐（与我们最相关）
| 项目 | 内容 |
|------|------|
| **问题** | MiniOneRec 只优化单目标（点击/购买），实际场景需同时优化多个目标 |
| **方案** | 结合 IVR 多任务经验（purchase + atc + open），设计多目标 reward 函数，在生成式框架下做多目标 RL |
| **核心问题** | 生成式框架下如何设计多任务 reward？多目标 GRPO 的训练稳定性如何保证？ |
| **数据集** | IVR（ivr_sample_v16），复用已有预处理 |
| **对比基线** | 传统 MMoE/PLE（已有结果）/ 单目标 MiniOneRec |

#### P10-C · 广告 CVR/Purchase 预估的生成式迁移
| 项目 | 内容 |
|------|------|
| **问题** | MiniOneRec 基于电商推荐（Amazon），广告场景有本质差异（曝光→点击→购买漏斗、PCOC 校准需求） |
| **方案** | 将生成式推荐适配到广告 CVR/Purchase 预估，研究生成式 vs 判别式方法的效果差异 |
| **核心问题** | 生成式方法能否在 AUC 和 PCOC 上同时超越传统判别式模型？ |
| **数据集** | IVR（ivr_sample_v16）|
| **对比基线** | MMoE AUC=0.8478（已有结果）|

---

### P10 · IVR 自监督对比学习

| 项目 | 内容 |
|------|------|
| **核心问题** | DSP 场景下 CVR 预估模型训练不稳定，正样本极度稀疏（CVR < 12%），传统监督学习难以充分利用特征表示信息 |
| **方法** | 对比学习（Contrastive Learning）增强 CVR 预估模型的特征表示能力 |
| **对比方法** | Baseline（DeepFM）/ SimGCL（扰动对比）/ SupCon（监督对比）/ DomainCL（域对比）/ FeatureMask（特征掩码）/ DirectAU（Alignment+Uniformity）/ MoCo（动量对比）/ Hybrid（混合方法） |
| **数据集** | ivr_sample_v16_ctcvr（326万训练，123万测试，126个类别特征，business_type 作为域标签） |
| **当前状态** | 实验完成（SupCon 最优，AUC 0.8490） |
| **关键突破** | 解决 GPU 训练数值稳定性问题（DataLoader + Xavier init + Dropout + Grad clipping） |
| **最终结果** | SupCon: AUC 0.8490（vs Baseline 0.8484，+0.06‰），参数调优后 SupCon 达 0.8488（Temp=0.2, CL_Weight=0.1） |
| **分析** | 对比学习在当前数据集上增益有限，可能是由于 DeepFM 已经是相对有效的模型架构 |
| **代码路径** | `ivr_ssl_cvr/` |
| **目标会议** | KDD / WWW |

**实验设计：**
- **SimGCL**: 在 embedding 层添加扰动，同一样本的两个扰动视图作为正例
- **SupCon**: 同标签样本为正例，不同标签为负例  
- **DomainCL**: 同 business_type 为正例，不同 domain 为负例
- **FeatureMask**: 随机掩码部分特征，同一样本的不同掩码视图作为正例
- **DirectAU**: Alignment + Uniformity 损失函数

**技术挑战：**
- GPU 训练数值稳定性（已解决）
- 大规模 embedding table 优化
- 对比学习计算效率

---

### ~~P-1 · CVR 延迟反馈（FDAM）~~ ⛔ 暂停

| 项目 | 内容 |
|------|------|
| **暂停原因** | 全量数据下 FDAM 持续落后 ES-DFM（0.7289 vs 0.7402），epoch 调参无效 |
| **最佳结果** | ep=8，AUC=0.7304，仍低于 ES-DFM 1.14 千分点 |
| **数据集** | Criteo KDD 2014（1589万行） |
| **代码** | `/mnt/workspace/git_project/AutoResearchClaw/cvr-delayed-feedback-kdd2026/` |
