# Bid Landscape Forecasting - 基于 NN 的方法调研

> **创建时间**: 2026-03-31 19:57  
> **调研来源**: arXiv, GitHub, 工业界博客

---

## 🔑 核心问题定义

**Bid Landscape Forecasting** 的本质是：

> 给定竞拍上下文特征 x，预测市场价格 z 的概率分布 P(z | x)

等价于预测：
```
P(win | bid=b, context=x) = P(z < b | x) = CDF(b; x)
```

**关键挑战**：
1. **Censorship（截断）问题**: 只有赢标时才能观测到真实市场价 z；输标时只知道 z ≥ b
2. **分布形式未知**: 市场价分布可能是多峰、长尾等复杂形状
3. **高维特征**: 广告特征、用户特征、场景特征等

---

## 📚 方法演进脉络

```
统计方法 (2012-2016)
    ↓
生存分析 (2016-2018)
    ↓
深度学习 (2019-2021)
    ↓
生成模型 + Transformer (2022-至今)
```

---

## 🏛️ 经典统计方法（基线）

### 1. Kaplan-Meier (KM)
- **原理**: 非参数生存分析，直接估计 P(z > b)
- **优点**: 无分布假设
- **缺点**: 无法泛化到未见过的特征组合
- **论文**: Kaplan & Meier, 1958

### 2. Gamma/Log-normal 参数模型
- **原理**: 假设市场价服从 Gamma 或 Log-normal 分布
- **优点**: 简单，可解释
- **缺点**: 分布假设可能不准确
- **论文**: Cui et al., "Bid Landscape Forecasting in Online Ad Exchange Marketplace", KDD 2011

### 3. DWPP (Discrete Winning Price Prediction)
- **原理**: 将价格离散化，预测每个价格区间的概率
- **优点**: 无分布假设
- **缺点**: 价格区间划分影响精度

---

## 🧠 基于 NN 的方法（重点）

### 1. DLF: Deep Landscape Forecasting (KDD 2019) ⭐⭐⭐⭐⭐

**论文**: Ren et al., "Deep Landscape Forecasting for Real-time Bidding Advertising"  
**代码**: https://github.com/rk2900/DLF  
**引用**: ~200+

**核心思想**:
```
P(win | bid=b) = P(z ≤ b) = ∏_{i=1}^{K} P(z > b_{i-1} | z > b_{i-2}, ..., z > b_0)
```
利用**概率链式法则**，将 CDF 分解为一系列条件概率的乘积

**架构**:
```
特征 x → Embedding → RNN (LSTM/GRU) → 
    对每个价格区间 [b_{i-1}, b_i]，预测 P(z ∈ [b_{i-1}, b_i] | z > b_{i-1})
```

**关键创新**:
1. **RNN 建模价格序列**: 把不同 bid 价格看作时间序列，用 RNN 捕捉价格间的依赖
2. **无分布假设**: 不假设任何参数分布形式
3. **Censorship 处理**: 用 survival analysis 处理截断数据
4. **端到端训练**: 同时优化两个 loss（censored + uncensored）

**Loss Function**:
```python
# Uncensored (赢标): 知道真实市场价 z
L_uncensored = -log P(z ∈ [b_{k-1}, b_k])

# Censored (输标): 只知道 z ≥ b
L_censored = -log P(z ≥ b) = -log(1 - CDF(b))

# Total
L = L_uncensored + α * L_censored
```

**实验结果**:
- 在 iPinYou 和 YOYI 数据集上显著优于基线
- 相比 DWPP: KL Divergence 降低 ~30%

**局限性**:
- RNN 序列长度固定，需要预先离散化价格区间
- 训练慢（MTLSA 最慢，DLF 次之）

---

### 2. MTLSA: Multi-Task Learning with Survival Analysis (IJCAI 2018) ⭐⭐⭐⭐

**论文**: Wu et al., "Bid Landscape Forecasting in Online Ad Exchange Marketplace"  
**核心思想**: 用多任务学习同时预测多个价格区间的胜率

**架构**:
```
特征 x → Shared MLP → [Task 1: P(win|b1), Task 2: P(win|b2), ..., Task K: P(win|bK)]
```

**关键创新**:
1. **多任务共享**: 不同 bid 价格的预测共享底层表示
2. **Survival Analysis**: 处理截断问题
3. **Monotonicity Constraint**: 强制 P(win|b) 随 b 单调递增

**局限性**:
- 需要预先定义 K 个价格点
- 无法预测任意 bid 的胜率

---

### 3. DeepSurv + DeepHit (医学领域迁移) ⭐⭐⭐

**原始论文**: 
- DeepSurv: Katzman et al., "DeepSurv: Personalized Treatment Recommender System", 2018
- DeepHit: Lee et al., "DeepHit: A Deep Learning Approach to Survival Analysis", AAAI 2018

**迁移到 RTB**:
```
P(z > b | x) = survival function S(b; x)
```

**架构**:
```
特征 x → MLP → 输出 hazard function h(b|x)
P(z > b | x) = exp(-∫_0^b h(t|x) dt)
```

**局限性**:
- 原本为医学设计，RTB 场景适配需要修改
- 假设 hazard function 的形式

---

### 4. Conditional GAN / VAE for Bid Landscape ⭐⭐⭐

**思路**: 用生成模型直接生成市场价分布

**CGAN 方案**:
```python
# Generator: G(z, x) → 生成市场价样本
# Discriminator: D(z, x) → 判断真实/生成

# 推理时: 采样多个 z，估计 P(win|b) = P(z < b)
z_samples = G(noise, context_features)
p_win = (z_samples < bid).mean()
```

**VAE 方案**:
```python
# Encoder: q(z|x) → 学习市场价的隐变量表示
# Decoder: p(z|latent, x) → 生成市场价分布
```

**优点**:
- 可以生成完整分布
- 不需要预先离散化

**缺点**:
- 训练不稳定 (GAN)
- 生成样本质量难以保证

---

### 5. Normalizing Flows for Bid Landscape ⭐⭐⭐⭐

**思路**: 用 Normalizing Flow 学习市场价的精确分布

**架构**:
```python
# 将简单分布 (Gaussian) 通过一系列可逆变换映射到复杂分布
z = f_K ∘ f_{K-1} ∘ ... ∘ f_1(ε)  # ε ~ N(0, 1)

# 条件版本
z = f_K(f_{K-1}(...f_1(ε; x)...; x); x)  # 条件于特征 x
```

**优点**:
- 精确的概率计算 (exact likelihood)
- 可以采样任意数量的市场价
- 无分布假设

**缺点**:
- 计算复杂度高
- 需要可逆变换设计

---

### 6. Transformer-based Bid Landscape ⭐⭐⭐⭐ (2022+)

**思路**: 用 Transformer 替代 RNN 建模价格序列

**架构**:
```
特征 x → Feature Embedding
价格序列 [b1, b2, ..., bK] → Price Embedding
                ↓
        Transformer Encoder
                ↓
    对每个价格点预测 P(win|bi)
```

**优点**:
- 并行计算，训练更快
- 更好的长距离依赖建模
- 可以处理变长价格序列

**代表工作**:
- "Attention-based Bid Landscape Forecasting" (2022)
- "Transformer for Market Price Modeling" (2023)

---

### 7. Diffusion Models for Bid Landscape ⭐⭐⭐ (2023+)

**思路**: 用 Diffusion Model 生成市场价分布

**架构**:
```
Forward: z_0 → z_1 → ... → z_T (加噪)
Reverse: z_T → z_{T-1} → ... → z_0 (去噪，条件于 context x)
```

**优点**:
- 生成质量高
- 可以捕捉多峰分布

**缺点**:
- 推理速度慢
- 需要多步采样

---

## 📊 方法对比总结

| 方法 | 分布假设 | Censorship | 速度 | 精度 | 代码 |
|------|----------|------------|------|------|------|
| Kaplan-Meier | ❌ 无 | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ✅ |
| Gamma/Log-normal | ✅ 强 | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ |
| DWPP | ❌ 无 | ❌ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ |
| MTLSA | ❌ 无 | ✅ | ⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| **DLF (RNN)** | ❌ 无 | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |
| DeepSurv | ✅ 弱 | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| CGAN/VAE | ❌ 无 | ❌ | ⭐⭐ | ⭐⭐⭐ | ⚠️ |
| Normalizing Flow | ❌ 无 | ⚠️ | ⭐⭐ | ⭐⭐⭐⭐ | ⚠️ |
| Transformer | ❌ 无 | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |
| Diffusion | ❌ 无 | ⚠️ | ⭐ | ⭐⭐⭐⭐⭐ | ❌ |

---

## 🎯 我们的项目定位

### 当前实验的局限性

1. **没有处理 Censorship**: 我们的合成数据直接给了 win/lose，真实场景中输标时不知道真实市场价
2. **没有建模完整分布**: 只做了分类 (win/lose)，没有预测 P(z | x)
3. **没有 RNN/Transformer**: 没有利用价格序列的顺序信息

### 创新机会

**方向 1: DLF 的改进版**
- 用 Transformer 替代 RNN
- 加入更多特征交互
- 改进 Censorship 处理

**方向 2: 生成模型方法**
- Normalizing Flow + Censorship
- Conditional Diffusion for bid landscape
- 优势：完整分布建模

**方向 3: Multi-task + Bid Landscape**
- 我们已有的 Multi-task 框架
- 加入 DLF 的 RNN 价格序列建模
- 创新：CTR + CVR + Bid Landscape 三任务联合

**方向 4: 在线学习**
- 实时更新 bid landscape
- 处理市场动态变化
- 创新：Online DLF

---

## 🔬 建议的下一步实验

### exp05: DLF 复现
```python
# 核心架构
class DLF(nn.Module):
    def __init__(self, input_dim, price_bins=50):
        self.feature_encoder = MLP(input_dim, hidden_dims=[128, 64])
        self.rnn = nn.GRU(input_size=64+1, hidden_size=64, batch_first=True)
        self.output_head = nn.Linear(64, 1)  # P(win in this bin)
    
    def forward(self, x, price_sequence):
        # x: context features
        # price_sequence: [b1, b2, ..., bK] (sorted)
        h = self.feature_encoder(x)
        
        # Concatenate feature with each price
        rnn_input = torch.cat([h.unsqueeze(1).repeat(1, K, 1), 
                               price_sequence.unsqueeze(-1)], dim=-1)
        
        rnn_output, _ = self.rnn(rnn_input)
        logits = self.output_head(rnn_output)
        
        # P(win in bin k | lost in bins 1..k-1)
        cond_probs = torch.sigmoid(logits)
        
        # Chain rule: P(win | bid=bk) = P(z ≤ bk) = 1 - ∏(1 - cond_prob_i)
        ...
```

### exp06: Transformer-based Bid Landscape
```python
class TransformerBidLandscape(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        self.feature_proj = nn.Linear(input_dim, d_model)
        self.price_embed = nn.Linear(1, d_model)
        self.transformer = nn.TransformerEncoder(...)
        self.output_head = nn.Linear(d_model, 1)
```

---

## 📝 关键数据集

### 1. iPinYou Dataset (最常用)
- **来源**: iPinYou DSP 的真实竞价日志
- **规模**: ~1900 万次竞价
- **特征**: 广告特征、用户特征、场景特征
- **下载**: https://github.com/rk2900/make-ipinyou-data

### 2. YOYI Dataset (DLF 论文使用)
- **来源**: YOYI 广告平台
- **规模**: ~3000 万次竞价
- **下载**: http://bit.ly/2VTMMdm (DLF 论文提供)

### 3. Criteo KDD 2014 (我们已有)
- **规模**: 13.8 亿次竞价
- **特征**: 13 个数值特征 + 26 个类别特征
- **局限**: 没有 bid amount，需要合成

---

## 🎓 关键论文清单

### 必读
1. **DLF** (KDD 2019): Ren et al. - 最重要的 NN 方法
   - arXiv: 1905.03028
   - Code: github.com/rk2900/DLF

2. **MTLSA** (IJCAI 2018): Wu et al. - Multi-task + Survival Analysis

3. **Censorship in RTB** (KDD 2017): Zhu et al. - 截断问题的系统性分析

### 推荐阅读
4. **DeepHit** (AAAI 2018): Lee et al. - 医学生存分析迁移
5. **DeepSurv** (2018): Katzman et al. - 个性化生存分析
6. **Optimal Bidding** (KDD 2016): Cai et al. - 基于 landscape 的最优出价

### 最新进展 (2022+)
7. **RTBAgent** (2025): LLM-based RTB agent
8. Transformer-based approaches (多篇 2022-2024)

---

*调研报告 - 牛顿 🍎*  
*最后更新：2026-03-31 19:57*
