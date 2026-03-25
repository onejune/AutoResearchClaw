# NODE (Neural Oblivious Decision Ensembles)

## 基本信息
- **论文**：[Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312)
- **会议/年份**：ICLR 2020
- **作者/机构**：Sergei Popov, Stanislav Morozov, Artem Babenko / Yandex Research

## 核心思路
NODE 将**遗忘决策树（Oblivious Decision Tree, ODT）**的结构嵌入到神经网络中，通过可微分的软分裂（soft splits）使整个模型端到端可训练。NODE 将多个 ODT 层叠加，每层包含大量并行的"软决策树"，最终对所有树的输出做平均，形成类似梯度提升树（GBDT）的集成效果，但完全可微。

## 方法细节

**遗忘决策树（ODT）：**
- 每层所有节点使用相同的分裂特征（"遗忘"性质）
- 深度为 d 的 ODT 有 $2^d$ 个叶节点

**可微分软分裂：**
$$h(x) = \text{entmax}_{1.5}\left(\frac{x - b}{\tau}\right)$$
其中 entmax 是 softmax 的稀疏版本，$b$ 为可学习分裂阈值，$\tau$ 为可学习温度。

**叶节点权重计算：**
对于深度 d 的树，叶节点权重为各层分裂结果的 Kronecker 积：
$$\mathbf{c} = \bigotimes_{i=1}^{d} h_i(x)$$

**最终输出：**
$$\text{NODE}(x) = \frac{1}{T} \sum_{t=1}^{T} \mathbf{R}_t^T \mathbf{c}_t$$
其中 $T$ 为树的数量，$\mathbf{R}_t$ 为叶节点响应向量（可学习）。

**数值特征处理：**
NODE 直接使用归一化后的数值特征，通过可学习的分裂阈值自动学习特征的分段结构。

## 报告效果
- **数据集**：Epsilon、YearPrediction、Higgs、Microsoft LETOR、Yahoo LETOR、Click
- Higgs AUC：0.8912（优于 XGBoost 的 0.8852）
- Click（CTR 预估）AUC：0.7697（优于 XGBoost 的 0.7688）
- 在多个大规模数据集上优于 XGBoost、CatBoost 等 GBDT 方法

## 优点
- 结合了决策树的归纳偏置和神经网络的可微性
- 可以使用 GPU 加速，训练速度快
- 天然支持特征重要性分析
- 在数值特征为主的数据集上效果优异

## 缺点 / 局限性
- 对类别特征处理较弱，需要预先 embedding
- 模型较大（大量并行树），内存占用高
- 超参数（树深度、树数量）对效果影响显著
- 在高维稀疏特征场景（如 CTR 预估）效果不如 FM 类方法
- 训练时间随树数量线性增长
