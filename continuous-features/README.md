# 连续特征处理方法对比实验

基于 Criteo 数据集，对比多种连续特征处理方式对 CTR 预估效果的影响。

## 实验目标

探究不同连续特征处理方法（从简单标量到复杂 Transformer）对 CTR 预估 AUC 的影响，量化各方法的效果-效率权衡。

## 数据集

- **Criteo Display Advertising Dataset**
- 数据路径：`/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_dataset/data.txt`
- 规模：1589 万行，13 个连续特征（I1-I13）+ 26 个类别特征（C1-C26）
- 实验采样：100 万行（前 80% 训练，后 20% 测试）

## 对比方法

| 编码器 | 描述 | 参考 |
|--------|------|------|
| ScalarEncoder | 直接使用标量值（baseline） | - |
| BucketEncoder | 等频分桶 + embedding lookup | 传统方法 |
| AutoDisEncoder | 可学习软分桶 + 元 embedding | KDD 2021 |
| NumericEmbedding | 每特征独立 MLP 映射 | NeurIPS 2022 |
| FTTransformer | 特征 token 化 + Transformer | NeurIPS 2021 |
| PeriodicEncoder | sin/cos 周期函数编码 | NeurIPS 2021/2022 |

## 目录结构

```
exp_continuous_features/
├── config.py            # 实验配置
├── data.py              # 数据加载与预处理
├── feature_encoders.py  # 各种连续特征处理方法
├── models.py            # DeepFM backbone
├── trainer.py           # 训练器（Adam + BCE + Early Stopping）
├── evaluate.py          # AUC 评估
├── main.py              # 主入口，跑所有方法输出对比表
├── results/             # 实验结果
│   └── comparison.txt   # 对比表
├── survey/              # 文献综述
│   ├── README.md        # 综述总览
│   ├── autodis.md
│   ├── fives.md
│   ├── ft_transformer.md
│   ├── tabnet.md
│   ├── node.md
│   ├── numeric_embedding.md
│   └── periodic_activations.md
└── README.md
```

## 快速开始

```bash
cd /mnt/workspace/open_research/autoresearch/exp_continuous_features

# 跑所有方法对比
python main.py

# 跑单个方法（修改 config.py 中的 encoder 字段）
python -c "
from config import Config
from data import get_dataloaders
from feature_encoders import build_encoder
from models import DeepFM
from trainer import Trainer
import torch

config = Config(encoder='autodis', epochs=3)
train_loader, test_loader, dataset = get_dataloaders(config)
encoder = build_encoder(config)
model = DeepFM(encoder, [config.cat_vocab_size]*26)
trainer = Trainer(model, config, torch.device('cpu'))
auc, t = trainer.fit(train_loader, test_loader)
print(f'AUC: {auc:.4f}')
"
```

## 模型架构

**DeepFM backbone：**
- FM 部分：一阶线性项 + 二阶特征交叉
- Deep 部分：MLP(256 → 128 → 64 → 1)，含 BatchNorm + Dropout
- 连续特征：通过各编码器处理后与类别 embedding 拼接

**训练配置：**
- 优化器：Adam，lr=1e-3
- 损失函数：BCE Loss
- Early Stopping：patience=2
- Batch Size：4096

## 实验配置

详见 `config.py`，主要参数：
- `sample_size`：采样行数（默认 100 万）
- `embedding_dim`：embedding 维度（默认 16）
- `epochs`：最大训练轮数（默认 3）
- `encoder`：编码器名称

## 文献综述

详见 `survey/` 目录，涵盖 AutoDis、FIVES、FT-Transformer、TabNet、NODE、Periodic Activations、Numeric Embedding 等方法。
