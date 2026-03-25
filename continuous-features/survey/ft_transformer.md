# FT-Transformer

## 基本信息
- **论文**：[Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)
- **会议/年份**：NeurIPS 2021
- **作者/机构**：Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko / Yandex Research

## 核心思路
FT-Transformer（Feature Tokenizer + Transformer）将表格数据中的每个特征（包括数值特征和类别特征）都转化为一个 token，然后用标准 Transformer 对所有 token 进行全局注意力建模。数值特征通过线性投影（标量乘以可学习权重向量再加偏置）转化为 token，类别特征通过 embedding lookup 转化为 token。这种方式使得 Transformer 能够自动学习任意阶的特征交互。

## 方法细节

**Feature Tokenizer（特征 token 化）：**

对于数值特征 $x_j \in \mathbb{R}$：
$$T_j = x_j \cdot \mathbf{w}_j + \mathbf{b}_j, \quad \mathbf{w}_j, \mathbf{b}_j \in \mathbb{R}^d$$

对于类别特征（one-hot $\mathbf{x}_j$）：
$$T_j = \mathbf{E}_j^T \mathbf{x}_j, \quad \mathbf{E}_j \in \mathbb{R}^{|C_j| \times d}$$

**Transformer 处理：**
- 将所有特征 token 拼接：$\mathbf{T} = [T_1; T_2; \ldots; T_m] \in \mathbb{R}^{m \times d}$
- 加入 [CLS] token（可选）
- 过 L 层 Transformer Encoder（多头注意力 + FFN）
- 取 [CLS] token 或所有 token 平均作为最终表示

**注意力机制：**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

## 报告效果
- **数据集**：11 个公开表格数据集（包括 California Housing、Adult、Covertype 等）
- 在多数数据集上优于或持平于 XGBoost、ResNet 等强基线
- 在 Higgs 数据集（分类）AUC：0.8912
- 首次证明 Transformer 在表格数据上可以与树模型竞争

## 优点
- 统一处理数值和类别特征，接口简洁
- Transformer 自动学习任意阶特征交互
- 无需手动特征工程
- 在多个基准数据集上取得 SOTA 效果

## 缺点 / 局限性
- 计算复杂度 O(n²)（n 为特征数），特征数多时较慢
- 参数量较大，训练需要更多数据
- 对超参数（d_model、层数、头数）较敏感
- 在特征数极多的工业级 CTR 场景（数百个特征）效率较低
