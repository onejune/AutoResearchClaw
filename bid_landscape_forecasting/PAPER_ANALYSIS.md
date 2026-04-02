# 论文分析与实验设计

**分析日期**: 2026-04-01  
**数据集**: Synthetic Bid Landscape (50 万样本，基于 IVR Sample v16 CTCVR)  
**评估指标**: AUC, RMSE, MAE, ECE (校准误差), CTR AUC (多任务)

---

## 📊 现有实验覆盖情况 (exp01-exp11)

| 实验 | 方法 | 对应论文/技术 |
|------|------|--------------|
| exp01 | Logistic Regression / XGBoost | 经典基线 |
| exp02 | MLP | 深度基线 |
| exp03 | Beta Regression | 分布建模 |
| exp04 | Multi-task (CTR+Win) | 多任务学习 |
| exp05 | **DLF (GRU)** | Ren et al. KDD 2019 |
| exp06 | **MTLSA** | IJCAI 2018 |
| exp07 | Censored Regression (Tobit) | 生存分析基线 |
| exp08 | Quantile NN | 分位数回归 |
| exp09 | Conformal Prediction | WWW 2023 Workshop |
| exp10 | **Deep Censored / Deep Cox PH** | 深度生存分析 |
| exp11 | Quantile Random Forest | 论文复现 |

---

## 📚 新论文分析与重复性检查

### 1. 《DeepWin: A Deep Recurrent Model for Real-Time Auction Win Rate Prediction》
**发表**: ACM TIST, 2026  
**核心**: LSTM + Attention 捕捉历史竞价序列时序依赖

#### 🔍 重复性分析
| 维度 | 现有实验 (exp05 DLF) | DeepWin | 重复度 |
|------|---------------------|---------|--------|
| 序列模型 | GRU | **LSTM** | ⚠️ 部分重复 |
| 注意力机制 | 无 | **Attention** | ✅ 新增 |
| 上下文特征 | 基础特征 | **用户画像 + 广告位 + 设备 + 地域** | ✅ 增强 |
| 出价嵌入 | 排序序列 | **Bid Embedding** | ⚠️ 类似 |
| 实时性 | 离线训练 | **毫秒级预测** | ✅ 新增优化 |

#### 📝 实验设计建议
**方案 A: 完全复现 (推荐)**
- 实现 LSTM + Attention 架构
- 添加 bid embedding 层
- 优化推理速度（模型剪枝/量化）

**方案 B: 增量改进**
- 在 exp05 基础上替换 GRU→LSTM
- 添加 Attention 机制
- 对比：DLF vs DeepWin

**结论**: ⚠️ **约 60% 重复**，建议实现完整 DeepWin 作为 exp12，重点对比 Attention 效果

---

### 2. 《Win Rate Estimation via Censored Data Modeling in RTB》
**发表**: WWW 2025  
**核心**: 生存分析框架，Cox 比例风险 / DeepHit 处理删失数据

#### 🔍 重复性分析
| 维度 | 现有实验 (exp10) | WWW 2025 Paper | 重复度 |
|------|-----------------|----------------|--------|
| 基础方法 | Deep Cox PH | **Cox PH + DeepHit** | ⚠️ 高度重复 |
| 删失处理 | 删失似然损失 | **边界信息利用** | ✅ 新增 |
| 特征工程 | 基础特征 | **低频/长尾流量增强** | ✅ 新增 |
| 评估指标 | AUC/RMSE/ECE | **+ Calibration on Tail** | ✅ 新增 |

#### 📝 实验设计建议
**关键差异点**:
1. **边界信息利用**: 输标时 `market_price > bid` 的不等式约束
2. **长尾流量分析**: 分频段评估（高频 vs 低频 traffic）
3. **DeepHit 实现**: 离散时间生存分析的深度学习版本

**实验设计**:
- exp13: DeepHit 复现 (离散时间生存分析)
- exp14: 边界约束增强 (Inequality-constrained Loss)
- 分组评估：按 impression frequency 分层

**结论**: ⚠️ **约 70% 重复**，但边界信息和长尾分析是新增价值点

---

### 3. 《Neural Bid Landscape Learning with Counterfactual Imputation》
**发表**: ICLR 2026 (Oral)  
**核心**: 生成式反事实模型 (GAN/VAE) + 联合训练

#### 🔍 重复性分析
| 维度 | 现有实验 | ICLR 2026 Paper | 重复度 |
|------|---------|-----------------|--------|
| 反事实生成 | ❌ 无 | **GAN/VAE Generator** | ✅ 全新 |
| 对抗训练 | ❌ 无 (exp16 已删) | **Discriminator + Generator** | ✅ 全新 |
| 端到端可微 | ❌ 分离训练 | **End-to-End** | ✅ 全新 |
| 公开数据集验证 | 合成数据 | **iPinYou / Yandex RTB** | ✅ 新增 |

#### 📝 实验设计建议
**实验设计**:
- exp15: Counterfactual Generator (VAE-based)
  - Input: context x, noise z
  - Output: market_price ~ P(price|x)
- exp16: Adversarial Co-Training
  - Generator: 生成反事实价格
  - Discriminator: 胜率预测 + 真假判别
  - Loss: L_supervised + λ·L_adversarial

**关键创新**:
1. 首次实现端到端可微
2. 在真实数据集上验证 (需下载 iPinYou/Yandex)

**结论**: ✅ **完全新颖**，与已删除的 exp14-16 思路相似但更系统化

---

### 4. 《LightWin: Efficient Win Rate Estimation for Large-Scale Industrial RTB Systems》
**来源**: KDD 2026 ADS (Alibaba)  
**核心**: 分桶 + 轻量 MLP，在线增量更新

#### 🔍 重复性分析
| 维度 | 现有实验 | LightWin | 重复度 |
|------|---------|----------|--------|
| 模型架构 | 标准 MLP/RNN | **分桶 + 轻量 MLP** | ✅ 新增 |
| 出价离散化 | 连续值 | **Bid Bucketing** | ✅ 新增 |
| 在线更新 | ❌ 离线训练 | **增量学习** | ✅ 全新 |
| 工业部署 | 学术研究 | **阿里妈妈生产系统** | ✅ 全新 |

#### 📝 实验设计建议
**实验设计**:
- exp17: Bid Bucketing + Lightweight MLP
  - 将 bid_amount 离散化为 N 个区间 (如 10/50/100 buckets)
  - 每个 bucket 独立子模型，共享底层 embedding
  - 模型大小：< 1MB (vs 原模型~50MB)
  
- exp18: Online Incremental Learning
  - 模拟流式数据场景
  - 每小时增量更新一次
  - 评估：稳定性 (CPM 波动) + 适应性 (概念漂移)

**评估指标扩展**:
- 推理延迟 (ms/sample)
- 模型大小 (MB)
- 内存占用
- 增量更新后的性能保持率

**结论**: ✅ **完全新颖**，工业实践视角，补充学术研究的不足

---

## 🎯 最终实验规划

### 优先级排序

| 优先级 | 实验编号 | 论文 | 新颖度 | 工作量 | 预期价值 |
|--------|---------|------|--------|--------|----------|
| **P0** | exp12 | DeepWin (TIST 2026) | ⭐⭐⭐ | 中 | 时序建模 SOTA |
| **P0** | exp13-14 | Censored Data (WWW 2025) | ⭐⭐ | 中 | 删失数据处理增强 |
| **P0** | exp15-16 | Counterfactual (ICLR 2026) | ⭐⭐⭐⭐⭐ | 高 | 因果推断前沿 |
| **P1** | exp17-18 | LightWin (KDD 2026) | ⭐⭐⭐⭐ | 中 | 工业落地参考 |

### 实验分组

**Group 1: 时序建模增强 (exp12)**
- DeepWin: LSTM + Attention + Bid Embedding
- 对比：exp05 (DLF-GRU) vs DeepWin

**Group 2: 删失数据深化 (exp13-14)**
- exp13: DeepHit (离散时间生存分析)
- exp14: 边界约束损失 + 长尾流量分析

**Group 3: 反事实因果推断 (exp15-16)**
- exp15: VAE-based Counterfactual Generator
- exp16: Adversarial Co-Training (Generator + Discriminator)

**Group 4: 工业效率优化 (exp17-18)**
- exp17: Bid Bucketing + Lightweight MLP
- exp18: Online Incremental Learning

---

## 📋 实验执行清单

### 通用准备
- [ ] 统一数据加载接口 (`data_loader.py`)
- [ ] 统一评估函数 (`metrics.py`: AUC, RMSE, ECE, Calibration Curve)
- [ ] 统一结果保存格式 (`results/expXX_*.json`, `results/expXX_*.md`)

### 实验记录模板
每个实验脚本头部添加：
```python
"""
实验 XX: [实验名称]

参考论文:
- [论文标题], [会议/期刊], [年份]
- DOI/arXiv: [链接]

核心思想:
[2-3 句话总结]

与现有实验的区别:
- vs expYY: [差异点 1]
- vs expZZ: [差异点 2]

实现细节:
[关键超参数、架构选择等]
"""
```

### 结果分析模板
每个实验生成 `results/expXX_analysis.md`:
```markdown
# expXX 实验分析

## 论文回顾
[论文核心贡献]

## 实验设置
- 数据集：...
- 超参数：...
- 训练时间：...

## 结果对比
| 方法 | AUC | RMSE | ECE | 备注 |
|------|-----|------|-----|------|
| expXX | ... | ... | ... | 本文 |
| expYY | ... | ... | ... | 基线 |

## 关键发现
1. ...
2. ...

## 局限性
1. ...

## 未来方向
1. ...
```

---

## 🚀 下一步行动

1. **创建实验 12: DeepWin** (LSTM + Attention)
2. **创建实验 13: DeepHit** (离散生存分析)
3. **创建实验 14: Censored with Bounds** (边界约束)
4. **创建实验 15: Counterfactual VAE**
5. **创建实验 16: Adversarial Co-Training**
6. **创建实验 17: LightWin Bucketing**
7. **创建实验 18: Online Learning**

所有实验使用统一的数据集和评估指标，确保可比性。

---

*Last updated: 2026-04-01*
