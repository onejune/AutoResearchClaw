# Ali-CCP 多任务 CVR 建模框架

基于 PyTorch 实现的多任务学习实验框架，支持 Ali-CCP 数据集，包含四个经典模型的端到端联合训练。

---

## 目录结构

```
exp_multitask/
├── config.py    # 实验配置（dataclass）
├── data.py      # 数据加载与预处理（支持 Ali-CCP 格式 + 合成数据）
├── models.py    # 四个 PyTorch 模型
├── trainer.py   # 训练器（统一训练/评估循环）
├── evaluate.py  # 评估指标（AUC）
├── main.py      # 实验入口，依次跑四个模型输出对比表
└── README.md    # 本文件
```

---

## 快速开始

### 1. 使用合成数据（无需下载数据集）

```bash
cd exp_multitask
python main.py
```

默认使用 10 万合成样本，5 个 epoch，约 2-5 分钟完成。

### 2. 使用真实 Ali-CCP 数据

将 Ali-CCP 数据放到某目录，例如 `/data/ali-ccp/`，目录下包含：
- `common_features_train.csv`
- `common_features_test.csv`
- `sample_skeleton_train.csv`
- `sample_skeleton_test.csv`

然后修改 `main.py` 中的 `config.data_dir`：

```python
config = Config(
    data_dir = "/data/ali-ccp/",
    sample_size = 500_000,   # None = 全量
    ...
)
```

---

## 模型说明

| 模型 | 核心思路 | 损失函数 |
|------|----------|----------|
| **SharedBottom** | 共享底层 MLP，分叉出 CTR/CVR 两个 tower | BCE(CTR) + BCE(CVR) |
| **ESMM** | 在整体曝光空间建模，p_ctcvr = p_ctr × p_cvr | BCE(CTR) + BCE(CTCVR) |
| **MMoE** | K 个专家 + 任务专属 Gate，软路由特征 | BCE(CTR) + BCE(CVR) |
| **ESCM2** | ESMM + 反事实正则化，惩罚未点击样本的高 CVR 预测 | BCE(CTR) + BCE(CTCVR) + λ·L_CR |

---

## 配置参数

编辑 `config.py` 或在 `main.py` 中直接修改 `Config(...)` 实例：

```python
Config(
    data_dir    = "",          # 空 = 合成数据
    sample_size = 500_000,     # 采样数，None = 全量
    batch_size  = 4096,
    model_name  = "esmm",      # shared_bottom / esmm / mmoe / escm2
    embedding_dim = 16,
    mlp_dims    = (256, 128, 64),
    dropout     = 0.1,
    n_experts   = 4,           # MMoE 专家数
    escm2_lambda = 0.1,        # ESCM2 反事实正则化强度
    epochs      = 5,
    lr          = 1e-3,
    seed        = 42,
    early_stopping_patience = 3,
)
```

---

## 数据格式（Ali-CCP）

**sample_skeleton_train.csv**（每行一个样本）：
```
sample_id, ctr_label, cvr_label, [其他列...]
```
- `ctr_label`：是否点击（1=点击）
- `cvr_label`：点击后是否购买（1=购买）
- `ctcvr_label` 自动计算为 `ctr_label × cvr_label`

**common_features_train.csv**（Long format，多行对应一个 sample）：
```
sample_id, feature_name, feature_value
```

---

## 合成数据规格

| 参数 | 值 |
|------|----|
| 用户数 | 10,000 |
| Item 数 | 50,000 |
| 类目数 | 100 |
| 特征 | user_id, item_id, category_id, hour, day_of_week, user_click_count, item_ctr |
| CTR 正样本率 | ~5% |
| CVR 正样本率（点击中） | ~20% |

---

## 依赖

```
torch >= 1.10
numpy
pandas
scikit-learn
```

安装：
```bash
pip install torch numpy pandas scikit-learn
```
