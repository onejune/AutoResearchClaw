# exp005: Domain Adversarial Neural Network (DANN)

**日期**: 2026-04-01  
**状态**: 🔄 进行中

## 目标

使用 Domain Adversarial Training 对齐训练集和 OOD 域的特征分布，重点解决 bt_16 分布漂移问题。

## 背景

bt_16 的关键问题：
- CTR 66% vs 训练集 42%（标签分布漂移）
- offerid 完全不同（bt_16 100% 是 offerid=37）
- is_reward_ad 反转（bt_16 有 61% 为 1 vs 训练集 18%）
- 9 个特征有 OOV 值

## 方法

DANN 通过梯度反转层（GRL）实现域对抗训练：
1. 特征提取器学习域不变特征
2. 域分类器预测样本来自哪个域（被 GRL 反转梯度）
3. 标签分类器进行 CTR 预估

### 网络结构

```
Input → Embedding → Feature Extractor (DNN) → [Label Classifier → CTR 预测]
                                     ↘
                                      → [Domain Classifier → 域分类]
                                         (Gradient Reversal Layer)
```

### 损失函数

```
L_total = L_label + λ * L_domain
```

其中 GRL 在反向传播时反转梯度，使特征提取器学习域不变特征。

## 实验设计

### 域划分
- **源域（Source）**：训练集 BT=[0,1,2,3,4,5,6,11]
- **目标域（Target）**：测试集 BT=[7,8,10,13,16]（只用特征，不用标签）

### 超参数
- λ 域对抗权重：[0.0, 0.1, 1.0, 10.0]
- GRL 渐进系数：从 0 线性增长到 1
- 其他同 baseline：lr=5e-5, batch_size=512, epochs=1

## 假设

- **H1**: DANN 能减小特征分布差异，提升 OOD AUC
- **H2**: λ 越大，域对齐越强，但可能损害标签预测
- **H3**: DANN 对 bt_16 有特殊改善（因为分布漂移最大）
