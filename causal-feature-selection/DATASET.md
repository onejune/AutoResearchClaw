# 数据集文档

## IVR v16 CTCVR Sample - 主数据集 ✅ 推荐使用

### 基本信息
- **路径**: `/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/`
- **状态**: ⭐ **已编码完成，开箱即用**
- **划分**: train / test (已分好)
- **特征处理**: 所有特征都已转为类别索引 (int32)，无需额外处理

### 数据结构

**训练集**:
- **样本数**: 3,265,331 (单文件 ~200MB)
- **特征数**: **126 个** (全部类别特征)
- **标签**: 
  - `click_label`: CTR 标签，正样本率 **41.49%**
  - `ctcvr_label`: CTCVR 标签，正样本率 **11.37%**

**测试集**: (待探查)

### 特征列表 (126 个)

#### 1. ID 类特征 (高维稀疏)
- `deviceid`: vocab_size = 2,383,974 (最高维!)
- `adid`: 11,637
- `campaignid`: 4,441
- `offerid`: ?
- `bundle`: 30,648
- `city`: 42,856
- `creativeid`: (未包含在最终 126 维中)

#### 2. 业务元数据 (低维)
- `business_type`: **20 个类别** (实际出现 19 个)
- `adx`: 46
- `country`: 47
- `os`: ?
- `carrier`: 8,191
- `language`: ?
- `imptype`: ?

#### 3. 用户行为统计 (中等维度，已 bucket 化)
- **duf_inner_dev_pkg_***: 内网设备 - 应用包行为
  - atc (add-to-cart): 1d~180d 多个时间窗口
  - pur (purchase): 购买行为
  - cv (conversion): 转化
  - imp (impression): 曝光
  - open: 打开
  - re (revenue): 收入
  - bucket 版本：分桶后的离散化版本
- **duf_outer_dev_pkg_***: 外网行为
- **huf_deviceid_demand_pkgname_***: 超短期实时特征 (1h/3h/12h/24h)

#### 4. 设备/广告属性
- `adsize`: 4,333
- `devicetype`: 9
- `make`: ?
- `model`: ?
- `osv`: ?
- `connectiontype`: 10
- `subcategory_id`: ?

### business_type 分布 (训练集)

| bt_id | 样本数 | 占比 |
|-------|--------|------|
| 1 | 1,097,581 | 33.6% |
| 11 | 488,377 | 15.0% |
| 6 | 315,616 | 9.7% |
| 4 | 300,593 | 9.2% |
| 2 | 278,231 | 8.5% |
| 0 | 276,360 | 8.5% |
| 3 | 183,045 | 5.6% |
| 8 | 119,026 | 3.6% |
| 5 | 106,003 | 3.2% |
| 10 | 50,987 | 1.6% |
| ...其他 9 个 | - | ~11.5% |

**总计**: 19 个业务类型，长尾分布

### 配套文件

```
ivr_sample_v16_ctcvr_sample/
├── encoders.pkl          # 编码器 (原始值 → 索引映射)
├── meta.json             # 元数据 (特征名、标签名)
├── vocab_sizes.json      # 每个特征的 vocab_size
├── train/
│   └── part-*.parquet    # 训练集
└── test/
    └── part-*.parquet    # 测试集
```

### 优势 ⭐

- ✅ **开箱即用**: 特征已编码，无需预处理
- ✅ **特征精简**: 从 1242 维降到 125 维（排除 `deviceid`，避免用户 ID 泄漏）
- ✅ **标签友好**: CTR 41.5% / CTCVR 11.4%，远好于原始数据
- ✅ **划分明确**: train/test 已分好
- ✅ **多任务**: 同时有 click 和 ctcvr 标签
- ✅ **跨域实验**: 19 个 business_type
- ✅ **数据量适中**: 326 万样本，单机可训练

### 使用示例

```python
import pandas as pd
import json

# 加载元数据
with open('/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/meta.json') as f:
    meta = json.load(f)

feature_cols = meta['feature_cols']  # 126 个特征名
label_cols = meta['label_cols']      # ['click_label', 'ctcvr_label']

# 加载 vocab_size
with open('/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/vocab_sizes.json') as f:
    vocab_sizes = json.load(f)

# 构建 feature_config (用于模型)
feature_config = {feat: vocab_sizes[feat] for feat in feature_cols}

# 加载数据
train_df = pd.read_parquet('/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/part-00000-*.parquet')

# 提取特征和标签
X = {feat: train_df[feat].values for feat in feature_cols}
y_click = train_df['click_label'].values
y_ctcvr = train_df['ctcvr_label'].values
```

---

## 原始 IVR v16 (备选，不推荐)

> 见项目根目录的原始数据探查记录，这里不再赘述。  
> **结论**: 用上面的预处理好版本即可，无需碰原始 3.2TB 数据。

---

## Criteo KDD 2014 (公开基准)

- **路径**: `/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_dataset/data.txt`
- **用途**: 论文对比时的公开数据集 baseline
- **特点**: 13 稠密 + 26 稀疏，4500 万样本，CTR ~13%

---

## 实验数据划分策略

### 跨域 OOD (Domain-based)

利用 `business_type` 构造:

```python
# Source domains (训练)
train_bts = [0, 1, 2, 3, 4, 5, 6, 11]  # Top 8, ~90% 数据

# Target domains (测试 - OOD)
test_bts = [7, 8, 10, 13, 16]  # 长尾 5 个
```

或者留一法:
```python
# Leave-one-out
for holdout_bt in range(19):
    train_bts = [bt for bt in range(19) if bt != holdout_bt]
    test_bts = [holdout_bt]
    # 训练 + 评估
```

### IRM 多环境构造

```python
environments = [
    {"name": "bt_group_1", "business_types": [0, 1, 2, 3]},
    {"name": "bt_group_2", "business_types": [4, 5, 6, 11]},
    {"name": "bt_group_3", "business_types": [7, 8, 10]},
    {"name": "bt_longtail", "business_types": [9, 12, 13, 14, 15, 16, 17, 18]},
]
```

---

## 注意事项

1. **deviceid 维度极高** (238 万)，embedding 层会很大，注意显存
2. **建议采样**: 先用 10%-50% 数据快速验证，再全量训练
3. **标签选择**: 
   - Phase 0-2: 用 `click_label` (CTR 任务，正样本多)
   - Phase 3-4: 可用 `ctcvr_label` (更难的转化任务)
4. **business_type 本身也是特征**: 可以做 ablation 看去掉后效果下降多少

---

**最后更新**: 2026-04-01  
**维护者**: 牛顿 🍎
