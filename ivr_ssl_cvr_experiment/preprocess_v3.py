#!/usr/bin/env python3
"""
预处理 ivr_sample_v16_ctcvr 数据集 → data_v3
- 采样 30%（训练+测试集均采样）
- 目标 7 个 business_type
- 标签：ctcvr_label
- Hash 编码所有特征
- 生成 train/val/user_id/business_type_id
"""
import os, sys, pickle, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

PROJECT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr'
DATA_DIR = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr'
OUT_DIR = f'{PROJECT_DIR}/data_v3'
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
SAMPLE_RATIO = 0.3
TARGET_BT = ['shein', 'aecps', 'aedsp', 'aerta', 'shopee_cps', 'lazada_cps', 'lazada_rta']

# 要排除的列
EXCLUDE_COLS = ['ctcvr_label', 'click_label', 'business_type', 'deviceid', 'label', 'business_type_id', 'user_id']


def load_and_sample(path, sample_ratio):
    """读取所有 parquet 文件，采样后合并"""
    print(f"读取 {path} ...")
    all_dfs = []
    for f in sorted(os.listdir(path)):
        if f.endswith('.parquet') and not f.startswith('_'):
            df = pd.read_parquet(os.path.join(path, f))
            # 采样
            df = df.sample(frac=sample_ratio, random_state=RANDOM_SEED)
            all_dfs.append(df)
            print(f"  {f}: {len(df):,} (采样后)")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  合并后: {len(combined):,}")
    return combined


def encode_features(df, exclude_cols):
    """Hash 编码所有特征列"""
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"特征数: {len(feature_cols)}")
    
    vocab_sizes = {}
    for col in tqdm(feature_cols, desc='Hash encoding'):
        if df[col].dtype == 'object' or str(df[col].dtype).startswith('string'):
            # 字符串列：先转为 str 再 hash
            codes = pd.util.hash_pandas_object(df[col].fillna('').astype(str)) % 100000
        else:
            # 数值列：直接 hash
            codes = pd.util.hash_pandas_object(df[col].fillna(0)) % 100000
        df[col] = codes.astype(np.int64)
        vocab_sizes[col] = int(df[col].max()) + 1
    
    return feature_cols, vocab_sizes


def main():
    np.random.seed(RANDOM_SEED)
    
    # 1. 加载并采样
    print("=" * 60)
    print("Step 1: 加载并采样数据")
    print("=" * 60)
    train_raw = load_and_sample(f'{DATA_DIR}/train', SAMPLE_RATIO)
    test_raw = load_and_sample(f'{DATA_DIR}/test', SAMPLE_RATIO)
    
    # 2. 过滤目标 BT
    print("\n" + "=" * 60)
    print("Step 2: 过滤目标 business_type")
    print("=" * 60)
    train_raw = train_raw[train_raw['business_type'].isin(TARGET_BT)].reset_index(drop=True)
    test_raw = test_raw[test_raw['business_type'].isin(TARGET_BT)].reset_index(drop=True)
    print(f"训练集（目标BT）: {len(train_raw):,}")
    print(f"测试集（目标BT）: {len(test_raw):,}")
    
    # 3. 编码标签
    train_raw['label'] = train_raw['ctcvr_label'].astype(np.int64)
    test_raw['label'] = test_raw['ctcvr_label'].astype(np.int64)
    
    # 4. 编码 business_type
    print("\n" + "=" * 60)
    print("Step 3: 编码 business_type")
    print("=" * 60)
    bt_encoder = LabelEncoder()
    train_raw['business_type_id'] = bt_encoder.fit_transform(train_raw['business_type'])
    # 测试集用同样的 encoder
    test_raw['business_type_id'] = test_raw['business_type'].apply(
        lambda x: bt_encoder.transform([x])[0] if x in bt_encoder.classes_ else -1
    )
    n_domains = len(bt_encoder.classes_)
    print(f"Business type: {list(bt_encoder.classes_)}")
    print(f"Domains: {n_domains}")
    
    # 5. 编码 deviceid → user_id
    print("\n" + "=" * 60)
    print("Step 4: 编码 deviceid → user_id")
    print("=" * 60)
    # 合并 train+test 的 deviceid 做统一编码，确保 user_id 一致
    all_deviceids = pd.concat([train_raw['deviceid'], test_raw['deviceid']], ignore_index=True).astype(str)
    uid_codes = pd.util.hash_pandas_object(all_deviceids) % 500000
    
    n_train = len(train_raw)
    train_raw['user_id'] = uid_codes[:n_train].values.astype(np.int64)
    test_raw['user_id'] = uid_codes[n_train:].values.astype(np.int64)
    n_users = int(uid_codes.max()) + 1
    print(f"User vocab size: {n_users}")
    print(f"Train 唯一用户: {train_raw['user_id'].nunique():,}")
    print(f"Test 唯一用户: {test_raw['user_id'].nunique():,}")
    
    # 6. Hash 编码特征
    print("\n" + "=" * 60)
    print("Step 5: Hash 编码特征")
    print("=" * 60)
    exclude = EXCLUDE_COLS + ['label']
    train_features, vocab_sizes = encode_features(train_raw, exclude)
    test_features, _ = encode_features(test_raw, exclude)
    # 确保特征一致
    assert train_features == test_features, "特征列不一致!"
    
    # 7. 整理列
    print("\n" + "=" * 60)
    print("Step 6: 保存数据")
    print("=" * 60)
    
    keep_cols = train_features + ['label', 'business_type_id', 'user_id']
    
    # 去重（原始 parquet 可能含重复列名）
    train_out = train_raw[keep_cols].loc[:, ~train_raw[keep_cols].columns.duplicated()]
    test_out = test_raw[keep_cols].loc[:, ~test_raw[keep_cols].columns.duplicated()]
    
    # 保存为 parquet（比 pkl 快且小）
    train_path = f'{OUT_DIR}/train.parquet'
    test_path = f'{OUT_DIR}/test.parquet'
    
    train_out.to_parquet(train_path, index=False, compression='snappy')
    test_out.to_parquet(test_path, index=False, compression='snappy')
    print(f"训练集: {len(train_out):,} 行 → {train_path}")
    print(f"测试集: {len(test_out):,} 行 → {test_path}")
    
    # 8. 保存元数据
    meta = {
        'features': train_features,
        'vocab_sizes': vocab_sizes,
        'bt_encoder': bt_encoder,
        'user_vocab_size': n_users,
        'n_domains': n_domains,
        'label_col': 'label',
        'version': 'v3',
        'sample_ratio': SAMPLE_RATIO,
        'target_business_types': TARGET_BT,
        'random_seed': RANDOM_SEED,
    }
    meta_path = f'{OUT_DIR}/meta.pkl'
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"元数据: {meta_path}")
    
    # 9. 打印统计
    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)
    print(f"\n训练集: {len(train_out):,} 行")
    print(f"  正样本: {train_out['label'].sum():,}  率={train_out['label'].mean():.4%}")
    print(f"  特征数: {len(train_features)}")
    print(f"\n测试集: {len(test_out):,} 行")
    print(f"  正样本: {test_out['label'].sum():,}  率={test_out['label'].mean():.4%}")
    print(f"\nper-BT 训练集:")
    for bt_id, name in enumerate(bt_encoder.classes_):
        mask = train_out['business_type_id'] == bt_id
        if mask.sum() == 0:
            continue
        n = mask.sum()
        n_pos = train_out.loc[mask, 'label'].sum()
        print(f"  {name}: 总={n:,} 正={int(n_pos):,} 率={n_pos/n:.4%}")
    print(f"\nper-BT 测试集:")
    for bt_id, name in enumerate(bt_encoder.classes_):
        mask = test_out['business_type_id'] == bt_id
        if mask.sum() == 0:
            continue
        n = mask.sum()
        n_pos = test_out.loc[mask, 'label'].sum()
        print(f"  {name}: 总={n:,} 正={int(n_pos):,} 率={n_pos/n:.4%}")
    
    # 用户分布
    train_uid = train_out.groupby('user_id').size()
    test_uid = test_out.groupby('user_id').size()
    print(f"\n用户分布:")
    print(f"  训练集: {train_uid.shape[0]:,} 唯一用户, 多样本用户: {(train_uid>=2).sum():,}")
    print(f"  测试集: {test_uid.shape[0]:,} 唯一用户, 多样本用户: {(test_uid>=2).sum():,}")
    
    # 10. 写 README
    readme = f"""# IVR SSL CVR 数据集 v3

基于 ivr_sample_v16_ctcvr，采样 {SAMPLE_RATIO*100:.0f}%。

## 数据来源
- `{DATA_DIR}/`

## 采样策略
- 训练集 & 测试集各随机采样 {SAMPLE_RATIO*100:.0f}%
- 目标 business_type: {TARGET_BT}
- 标签: ctcvr_label
- deviceid → user_id (hash % 500000)

## 数据统计
- 训练集: {len(train_out):,} 行
- 测试集: {len(test_out):,} 行
- 特征数: {len(train_features)}
- User vocab: {n_users}

## 生成命令
```bash
python preprocess_v3.py
```
"""
    with open(f'{OUT_DIR}/README.md', 'w') as f:
        f.write(readme)
    
    print(f"\n✅ 完成! 输出目录: {OUT_DIR}")
    print(f"   文件: train.parquet, test.parquet, meta.pkl")


if __name__ == '__main__':
    main()