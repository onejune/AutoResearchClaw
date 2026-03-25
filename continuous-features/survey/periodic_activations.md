# Periodic Activations for Positional Encoding

## 基本信息
- **论文**：[On the Embeddings of Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556)（综述中包含 Periodic Embedding）；原始周期激活思想来自 [Implicit Neural Representations with Periodic Activation Functions (SIREN)](https://arxiv.org/abs/2006.09661)
- **会议/年份**：NeurIPS 2021（SIREN）；NeurIPS 2022（表格数据综述）
- **作者/机构**：Vincent Sitzmann 等（SIREN）；Yury Gorishniy 等（表格应用）

## 核心思路
周期性激活函数编码（Periodic Encoding）将标量数值特征通过可学习的正弦/余弦函数映射到高维空间。核心思想是：数值特征的周期性表示可以捕获不同频率下的特征模式，类似于傅里叶变换将信号分解为不同频率成分。与固定频率的位置编码（如 Transformer 的 sinusoidal PE）不同，这里的频率（权重 w）和相位（偏置 b）都是可学习的。

## 方法细节

**基本公式：**
对于标量特征 $x$，周期编码为：
$$\text{PE}(x) = [\sin(w_1 x + b_1), \cos(w_1 x + b_1), \ldots, \sin(w_K x + b_K), \cos(w_K x + b_K)]$$
其中 $w_k, b_k \in \mathbb{R}$ 为可学习参数，$K$ 为频率数量。

**输出维度：** $2K$（每个频率对应 sin 和 cos 两个分量）

**参数初始化：**
- $w_k \sim \mathcal{N}(0, \sigma^2)$，$\sigma$ 为超参数（通常取 1.0）
- $b_k \sim \mathcal{N}(0, \sigma^2)$

**与固定 PE 的区别：**
- 固定 PE（如 Transformer）：$w_k = 1/10000^{2k/d}$，不可学习
- 可学习 PE：$w_k, b_k$ 通过梯度下降优化，适应数据分布

**与 SIREN 的联系：**
SIREN 使用 $\sin(\omega_0 \mathbf{W} \mathbf{x} + \mathbf{b})$ 作为网络激活函数，周期编码可视为 SIREN 第一层的特殊情况（输入为标量）。

## 报告效果
- **数据集**：California Housing、Adult、Higgs、Covertype 等
- 在 Gorishniy et al. (2022) 综述中，Periodic Embedding 在多个数据集上优于 MLP 直接输入
- Higgs AUC：~0.8920（优于无 embedding 的 0.8912）
- 在需要捕获周期性模式的特征（如时间特征、角度特征）上效果尤为显著

## 优点
- 参数量极少（每个特征仅 2K 个参数）
- 天然捕获数值特征的周期性和多频率模式
- 可学习频率比固定频率更灵活
- 计算高效（仅需 sin/cos 运算）

## 缺点 / 局限性
- 对于单调递增/递减的特征（如年龄、收入），周期性假设可能不合适
- 频率数量 K 是超参数，需要调优
- 初始化方差 $\sigma$ 对训练稳定性有影响
- 输出维度 $2K$ 较大，可能引入冗余信息
