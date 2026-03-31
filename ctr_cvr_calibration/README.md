# CTR/CVR 预估校准研究

> **研究目的**: 探索深度学习模型在 CTR/CVR 预估中的概率校准方法  
> **创建日期**: 2026-03-31  
> **状态**: 🔄 项目初始化  
> **工具**: AutoResearchClaw

---

## 📋 研究背景

### 校准问题定义

**校准 (Calibration)**: 模型预测概率与真实概率的一致性

```
预测 CTR = 0.4 → 实际点击率应接近 40%
预测 CVR = 0.1 → 实际转化率应接近 10%
```

### 研究问题

1. **深度学习模型的校准性如何？**
   - AUC 高是否意味着校准好？
   - 不同模型架构的校准差异？

2. **哪些校准方法有效？**
   - Temperature Scaling
   - Isotonic Regression
   - Focal Loss
   - Label Smoothing

3. **校准与排序的关系？**
   - 校准是否影响 AUC？
   - 如何平衡校准与排序？

4. **多任务学习下的联合校准？**
   - CTR + CVR 联合校准策略
   - ESMM 的校准偏差问题

---

## 🎯 实验设计

### 核心假设

1. **H1**: 深度学习模型普遍存在过度自信 (overconfidence)
2. **H2**: Temperature Scaling 是最有效的后处理校准方法
3. **H3**: Focal Loss 可以在训练时改善校准
4. **H4**: 多任务联合校准优于独立校准

### 对比方法

| 方法 | 类型 | 说明 |
|------|------|------|
| **Baseline** | - | 无校准 |
| **Temperature Scaling** | 后处理 | 单参数调整 |
| **Isotonic Regression** | 后处理 | 保序回归 |
| **Histogram Binning** | 后处理 | 分箱校准 |
| **Focal Loss** | 训练时 | 不平衡损失 |
| **Label Smoothing** | 训练时 | 正则化 |

### 模型架构

| 模型 | 说明 |
|------|------|
| **MLP** | 基础多层感知机 |
| **DeepFM** | 因子分解机 + DNN |
| **DCNv2** | Deep & Cross Network |
| **AutoInt** | 自动特征交互 |

---

## 📊 数据集

### IVR Sample v16 CTCVR

| 属性 | 值 |
|------|-----|
| **训练集** | 3,265,331 样本 |
| **特征数** | 126 (全部类别特征) |
| **标签** | click_label, ctcvr_label |
| **点击率 (CTR)** | 41.49% |
| **CTCVR** | 11.37% |

### 数据路径

```
/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/
├── train/
│   └── part-00000-*.parquet
├── test/
├── encoders.pkl
├── vocab_sizes.json
└── meta.json
```

### 特征处理

**重要**: 所有特征都当成类别特征，不做任何处理
- 直接使用已编码的数值 ID
- 使用 vocab_sizes.json 中的词表大小

---

## 📁 项目结构

```
ctr_cvr_calibration/
├── config.researchclaw.yaml    # AutoResearchClaw 配置
├── README.md                   # 本文件
├── QUICKSTART.md               # 快速开始
├── data/
│   ├── dataset.py              # 数据加载
│   └── preprocess.py           # 数据预处理
├── models/
│   ├── mlp.py                  # MLP 模型
│   ├── deepfm.py               # DeepFM 模型
│   └── dcn.py                  # DCNv2 模型
├── calibration/
│   ├── methods.py              # 校准方法
│   ├── evaluation.py           # 校准评估
│   └── multitask.py            # 多任务校准
├── experiments/
│   ├── exp01_baseline.py       # 基线实验
│   ├── exp02_temperature.py    # Temperature Scaling
│   ├── exp03_isotonic.py       # Isotonic Regression
│   ├── exp04_focal_loss.py     # Focal Loss
│   └── exp05_multitask.py      # 多任务联合校准
├── results/                    # 实验结果
├── logs/                       # 训练日志
└── checkpoints/                # 模型检查点
```

---

## 🚀 快速开始

### 1. 使用 AutoResearchClaw 启动

```bash
cd /mnt/workspace/open_research/autoresearch/ctr_cvr_calibration

# 初始化
researchclaw init

# 运行全流程
researchclaw run --topic "CTR/CVR 预估校准研究" --auto-approve
```

### 2. 手动运行实验

```bash
# 运行基线实验
python experiments/exp01_baseline.py

# 运行 Temperature Scaling
python experiments/exp02_temperature.py

# 运行所有实验
bash scripts/run_all_experiments.sh
```

---

## 📈 核心指标

| 指标 | 定义 | 理想值 |
|------|------|--------|
| **ECE** | Expected Calibration Error | 0 |
| **MCE** | Maximum Calibration Error | 0 |
| **PCOC** | Predicted/Observed CTR | 1.0 |
| **AUC** | Area Under ROC Curve | 越高越好 |

---

## 📝 实验日志

### 2026-03-31 项目初始化

- [x] 创建项目目录
- [x] 配置 AutoResearchClaw
- [ ] 数据质量检查
- [ ] 实现代码
- [ ] 运行实验

---

## 📚 参考文献

1. Guo et al., 2017 - "On Calibration of Modern Neural Networks" (ICML)
2. Niculescu-Mizil & Caruana, 2005 - "Predicting Good Probabilities With Supervised Learning"
3. Mukhoti et al., 2020 - "Calibrating Deep Neural Networks using Focal Loss"
4. Ma et al., 2018 - "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rates"

---

*研究维护：牛顿 🍎*
